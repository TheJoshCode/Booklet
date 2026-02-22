#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <zmq.h>
#include <json-c/json.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/file.h>


#define MAX_WORKERS 16
#define MAX_TEXT_SIZE 65536
#define SOCKET_PREFIX "ipc:///tmp/tts_worker_"
#define CHUNK_OVERLAP 50

/* Worker ping retry parameters */
#define PING_MAX_RETRIES   120    /* 120 × 500 ms = 60 s max wait */
#define PING_RETRY_SLEEP_MS 500

#define CORE_ALLOC_FILE "/tmp/tts_core_allocation.json"

static int  g_allocated_cores[256];
static int  g_allocated_count = 0;
static char g_run_id_global[64];


typedef struct {
    int   id;
    pid_t pid;
    char  socket_addr[256];
    void *zmq_socket;
    int   busy;
    pthread_mutex_t lock;
} Worker;

typedef struct {
    char **chunks;
    int   *chunk_indices;
    int    count;
    int    capacity;
} ChunkList;

typedef struct {
    char  *text;
    char  *voice_path;
    char  *output_path;
    char  *runs_dir;
    char  *run_id;
    int    max_chunk_size;
    int    num_workers;
    int    precision;      /* 0=int8, 1=fp32 */
    float  temperature;
    int    lsd_steps;
} Config;

/* Global state for cleanup */
static Worker   g_workers[MAX_WORKERS];
static int      g_num_workers = 0;
static void    *g_zmq_context = NULL;
static volatile int g_running = 1;
static volatile int g_cleanup_done = 0;  /* guard against double-cleanup (signal + atexit) */

/* Global progress tracking */
static volatile int  g_completed_chunks = 0;
static volatile int  g_total_chunks     = 0;
static pthread_mutex_t g_progress_mutex = PTHREAD_MUTEX_INITIALIZER;
static time_t g_start_time = 0;

/* -----------------------------------------------------------------------
 * Dynamic work queue — replaces static chunk-group division.
 * Workers pull from this queue instead of having a pre-assigned range, so
 * fast workers automatically absorb chunks from slow workers.
 * --------------------------------------------------------------------- */
typedef struct {
    int             next_chunk;   /* index of next chunk to dispatch      */
    int             total;        /* total number of chunks                */
    pthread_mutex_t mu;
    pthread_cond_t  cv;
} WorkQueue;

static WorkQueue g_work_queue;

static void work_queue_init(WorkQueue *q, int total) {
    q->next_chunk = 0;
    q->total      = total;
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->cv, NULL);
}

/* Returns the next chunk index, or -1 if all done. */
static int work_queue_pop(WorkQueue *q) {
    pthread_mutex_lock(&q->mu);
    int idx = -1;
    if (q->next_chunk < q->total)
        idx = q->next_chunk++;
    pthread_mutex_unlock(&q->mu);
    return idx;
}

static void work_queue_destroy(WorkQueue *q) {
    pthread_mutex_destroy(&q->mu);
    pthread_cond_destroy(&q->cv);
}


int get_total_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (int)n;
}

int allocate_cores(const char *run_id, int requested) {
    int total = get_total_cores();

    int fd = open(CORE_ALLOC_FILE, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open core allocation file");
        return -1;
    }

    flock(fd, LOCK_EX);

    FILE *f = fdopen(fd, "r+");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    struct json_object *root;
    if (size > 0) {
        char *buf = malloc(size + 1);
        if (fread(buf, 1, size, f) != (size_t)size)
            perror("fread");
        buf[size] = 0;
        root = json_tokener_parse(buf);
        free(buf);
        if (!root) root = json_object_new_object();
    } else {
        root = json_object_new_object();
    }

    struct json_object *allocations;
    if (!json_object_object_get_ex(root, "allocations", &allocations)) {
        allocations = json_object_new_object();
        json_object_object_add(root, "allocations", allocations);
    }

    /* Evict stale entries: any run whose recorded PID is no longer alive. */
    {
        struct json_object *to_evict = json_object_new_array();
        json_object_object_foreach(allocations, key, val) {
            struct json_object *pid_obj;
            if (json_object_object_get_ex(val, "pid", &pid_obj)) {
                pid_t owner = (pid_t)json_object_get_int(pid_obj);
                if (owner > 0 && kill(owner, 0) != 0 && errno == ESRCH)
                    json_object_array_add(to_evict, json_object_new_string(key));
            }
        }
        int nevict = json_object_array_length(to_evict);
        for (int i = 0; i < nevict; i++) {
            const char *k = json_object_get_string(
                json_object_array_get_idx(to_evict, i));
            printf("ℹ Evicting stale core allocation for run '%s'\n", k);
            json_object_object_del(allocations, k);
        }
        json_object_put(to_evict);
    }

    int used[256] = {0};

    json_object_object_foreach(allocations, key2, val2) {
        (void)key2;
        struct json_object *cores_obj;
        if (!json_object_object_get_ex(val2, "cores", &cores_obj)) continue;
        int len = json_object_array_length(cores_obj);
        for (int i = 0; i < len; i++) {
            int core = json_object_get_int(json_object_array_get_idx(cores_obj, i));
            if (core >= 0 && core < 256)
                used[core] = 1;
        }
    }

    g_allocated_count = 0;
    struct json_object *my_cores = json_object_new_array();

    /* Phase 1: preferred cores (skip 1 and 2) */
    for (int i = 0; i < total && g_allocated_count < requested; i++) {
        if (i == 1 || i == 2) continue;
        if (!used[i]) {
            json_object_array_add(my_cores, json_object_new_int(i));
            g_allocated_cores[g_allocated_count++] = i;
        }
    }

    /* Phase 2: use cores 1 and 2 if still needed */
    for (int i = 0; i < total && g_allocated_count < requested; i++) {
        if (i != 1 && i != 2) continue;
        if (!used[i]) {
            json_object_array_add(my_cores, json_object_new_int(i));
            g_allocated_cores[g_allocated_count++] = i;
        }
    }

    if (g_allocated_count < requested)
        printf("⚠ Not enough free cores. Requested %d, got %d\n",
               requested, g_allocated_count);

    /* Store {pid, cores} so future runs can evict us if we crash. */
    struct json_object *my_entry = json_object_new_object();
    json_object_object_add(my_entry, "pid",   json_object_new_int((int)getpid()));
    json_object_object_add(my_entry, "cores", my_cores);
    json_object_object_add(allocations, run_id, my_entry);

    rewind(f);
    if (ftruncate(fd, 0) != 0)
        perror("ftruncate");

    const char *json_str =
        json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
    fwrite(json_str, 1, strlen(json_str), f);
    fflush(f);

    flock(fd, LOCK_UN);
    fclose(f);
    json_object_put(root);

    return g_allocated_count;
}

void release_cores(const char *run_id) {
    if (!run_id || run_id[0] == '\0') return;

    int fd = open(CORE_ALLOC_FILE, O_RDWR);
    if (fd < 0) return;

    flock(fd, LOCK_EX);

    FILE *f = fdopen(fd, "r+");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    if (size <= 0) {
        flock(fd, LOCK_UN);
        fclose(f);
        return;
    }

    char *buf = malloc(size + 1);
    if (fread(buf, 1, size, f) != (size_t)size)
        perror("fread failed");
    buf[size] = 0;

    struct json_object *root = json_tokener_parse(buf);
    free(buf);

    if (!root) {
        flock(fd, LOCK_UN);
        fclose(f);
        return;
    }

    struct json_object *allocations;
    if (json_object_object_get_ex(root, "allocations", &allocations))
        json_object_object_del(allocations, run_id);

    rewind(f);
    if (ftruncate(fd, 0) != 0)
        perror("ftruncate failed");

    const char *json_str =
        json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
    fwrite(json_str, 1, strlen(json_str), f);
    fflush(f);

    flock(fd, LOCK_UN);
    fclose(f);
    json_object_put(root);
}

void cleanup(void) {
    /* Guard: atexit + signal_handler both call cleanup(); only run once. */
    if (g_cleanup_done) return;
    g_cleanup_done = 1;

    printf("\nCleaning up workers...\n");
    for (int i = 0; i < g_num_workers; i++) {
        if (g_workers[i].zmq_socket) {
            zmq_close(g_workers[i].zmq_socket);
            g_workers[i].zmq_socket = NULL;
        }
        if (g_workers[i].pid > 0) {
            kill(g_workers[i].pid, SIGTERM);

            /* Wait up to 2 s for graceful exit, then SIGKILL. */
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            deadline.tv_sec += 2;

            int done = 0;
            while (!done) {
                int wstatus;
                pid_t r = waitpid(g_workers[i].pid, &wstatus, WNOHANG);
                if (r > 0) {
                    done = 1;
                } else if (r < 0) {
                    done = 1; /* already gone */
                } else {
                    struct timespec now;
                    clock_gettime(CLOCK_REALTIME, &now);
                    if (now.tv_sec > deadline.tv_sec ||
                        (now.tv_sec == deadline.tv_sec &&
                         now.tv_nsec >= deadline.tv_nsec)) {
                        kill(g_workers[i].pid, SIGKILL);
                        waitpid(g_workers[i].pid, NULL, 0);
                        done = 1;
                    } else {
                        usleep(50000); /* 50 ms poll */
                    }
                }
            }
            g_workers[i].pid = 0;
        }
        pthread_mutex_destroy(&g_workers[i].lock);
    }
    if (g_zmq_context) {
        zmq_ctx_term(g_zmq_context);
        g_zmq_context = NULL;
    }
    release_cores(g_run_id_global);
}

void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
    /* cleanup() is registered via atexit(); calling exit() is sufficient.
     * Do NOT call cleanup() here directly — that would cause double-cleanup
     * since exit() fires atexit handlers after this returns. */
    exit(0);
}

/* Format seconds into human-readable time */
void format_time(int seconds, char *buffer, size_t size) {
    if (seconds < 60)
        snprintf(buffer, size, "%ds", seconds);
    else if (seconds < 3600)
        snprintf(buffer, size, "%dm %ds", seconds / 60, seconds % 60);
    else
        snprintf(buffer, size, "%dh %dm", seconds / 3600, (seconds % 3600) / 60);
}

/* Display unified progress bar */
void display_progress_bar(void) {
    pthread_mutex_lock(&g_progress_mutex);
    int completed = g_completed_chunks;
    int total     = g_total_chunks;
    pthread_mutex_unlock(&g_progress_mutex);

    if (total == 0) return;

    int percent   = (completed * 100) / total;
    int bar_width = 40;
    int filled    = (completed * bar_width) / total;

    time_t now = time(NULL);
    double elapsed     = difftime(now, g_start_time);
    double eta_seconds = 0;
    if (completed > 0) {
        double avg = elapsed / completed;
        eta_seconds = avg * (total - completed);
    }

    char eta_str[32], elapsed_str[32];
    format_time((int)elapsed,     elapsed_str, sizeof(elapsed_str));
    format_time((int)eta_seconds, eta_str,     sizeof(eta_str));

    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled)       printf("█");
        else if (i == filled) printf("▒");
        else                  printf("░");
    }
    printf("] %d/%d (%d%%) | Elapsed: %s | ETA: %s    ",
           completed, total, percent,
           elapsed_str,
           eta_seconds > 0 ? eta_str : "calculating...");
    fflush(stdout);
}

/* Generate timestamp-based run ID */
void generate_run_id(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    snprintf(buffer, size, "%04d%02d%02d_%02d%02d%02d",
             tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
             tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);
}

/* Create directory recursively */
int mkdir_p(const char *path) {
    char tmp[1024];
    char *p = NULL;

    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
    return 0;
}

/* Find sentence boundary near position */
int find_sentence_boundary(const char *text, int target_pos, int text_len) {
    if (target_pos >= text_len) return text_len;

    int search_start = target_pos - CHUNK_OVERLAP;
    if (search_start < 0) search_start = 0;

    for (int i = target_pos; i > search_start; i--) {
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            if (i + 1 < text_len && (text[i+1] == ' ' || text[i+1] == '\n'))
                return i + 1;
        }
        if (text[i] == '\n' && i > search_start + 10)
            return i;
    }

    for (int i = target_pos; i > search_start; i--) {
        if (text[i] == ' ')
            return i;
    }

    return target_pos;
}

ChunkList* split_text(const char *text, int max_chunk_size) {
    ChunkList *list = malloc(sizeof(ChunkList));
    list->count    = 0;
    list->capacity = 16;
    list->chunks        = malloc(list->capacity * sizeof(char*));
    list->chunk_indices = malloc(list->capacity * sizeof(int));

    int text_len = (int)strlen(text);
    int pos = 0, chunk_num = 0;

    while (pos < text_len) {
        int end_pos = pos + max_chunk_size;

        if (end_pos >= text_len)
            end_pos = text_len;
        else
            end_pos = find_sentence_boundary(text, end_pos, text_len);

        int   chunk_len = end_pos - pos;
        char *chunk     = malloc(chunk_len + 1);
        strncpy(chunk, text + pos, chunk_len);
        chunk[chunk_len] = '\0';

        char *start = chunk;
        while (*start == ' ' || *start == '\n' || *start == '\t') start++;
        char *end = chunk + strlen(chunk) - 1;
        while (end > start && (*end == ' ' || *end == '\n' || *end == '\t')) *end-- = '\0';

        if (strlen(start) > 0) {
            if (list->count >= list->capacity) {
                list->capacity *= 2;
                list->chunks        = realloc(list->chunks,
                                              list->capacity * sizeof(char*));
                list->chunk_indices = realloc(list->chunk_indices,
                                              list->capacity * sizeof(int));
            }
            list->chunks[list->count]        = strdup(start);
            list->chunk_indices[list->count] = chunk_num;
            list->count++;
        }

        free(chunk);
        pos = end_pos;
        chunk_num++;
    }

    return list;
}

void free_chunks(ChunkList *list) {
    for (int i = 0; i < list->count; i++)
        free(list->chunks[i]);
    free(list->chunks);
    free(list->chunk_indices);
    free(list);
}

/* -----------------------------------------------------------------------
 * Worker startup
 *
 * OPTIMIZATION: replaced sleep(5) with a poll/retry loop.
 *   Old: unconditional sleep(5) before connecting — wasted up to 5 seconds
 *        even if all workers loaded in 2 s, and could fail if loading > 5 s.
 *   New: poll every PING_RETRY_SLEEP_MS ms, up to PING_MAX_RETRIES attempts.
 *        Workers become available as soon as they're ready.
 *
 * OPTIMIZATION: removed sched_setaffinity.
 *   Old: each Python worker pinned to exactly 1 physical core.
 *        PocketTTSOnnx configures intra_op_num_threads=4, but with affinity
 *        set to 1 core the OS can never schedule those threads elsewhere —
 *        effectively forcing single-core ONNX inference regardless of config.
 *   New: let the OS scheduler place threads freely.  ONNX can use all cores
 *        allocated to it via intra_op_num_threads.  If isolation is desired,
 *        give each worker a *set* of cores (num_cores / num_workers), not 1.
 * --------------------------------------------------------------------- */
int start_workers(Config *cfg) {
    g_zmq_context = zmq_ctx_new();

    /* Phase 1: Fork all workers */
    for (int i = 0; i < cfg->num_workers; i++) {
        Worker *w = &g_workers[i];
        w->id         = i;
        w->busy       = 0;
        w->zmq_socket = NULL;
        pthread_mutex_init(&w->lock, NULL);
        snprintf(w->socket_addr, sizeof(w->socket_addr),
                 "%s%d.sock", SOCKET_PREFIX, i);

        /* Remove old socket file */
        char path[256];
        snprintf(path, sizeof(path), "/tmp/tts_worker_%d.sock", i);
        unlink(path);

        pid_t pid = fork();
        if (pid == 0) {
            /* Child: exec the Python worker */
            char worker_id_s[16], temp_s[16], lsd_s[16];
            snprintf(worker_id_s, sizeof(worker_id_s), "%d", i);
            snprintf(temp_s,      sizeof(temp_s),      "%.2f", cfg->temperature);
            snprintf(lsd_s,       sizeof(lsd_s),       "%d", cfg->lsd_steps);

            execlp("uv", "uv",
                "run",
                "worker_instance.py",
                "--worker-id", worker_id_s,
                "--socket",    w->socket_addr,
                "--precision", cfg->precision ? "fp32" : "int8",
                "--temperature", temp_s,
                "--lsd-steps",  lsd_s,
                (char *)NULL);

            perror("Failed to exec worker");
            exit(1);

        } else if (pid > 0) {
            w->pid = pid;
            printf("Started worker %d (PID %d)\n", i, pid);

            /* REMOVED: sched_setaffinity — see notes above */

        } else {
            perror("fork failed");
            return -1;
        }
    }

    g_num_workers = cfg->num_workers;

    /* Phase 2: Poll until every worker responds to a ping.
     *
     * OPTIMIZATION: was sleep(5) followed by a single connect+ping attempt
     * per worker.  That's up to 5 s of dead time before any work starts.
     * Now we retry immediately and proceed as soon as each worker is live.
     */
    printf("Waiting for workers to initialize (polling)...\n");

    for (int i = 0; i < cfg->num_workers; i++) {
        Worker *w = &g_workers[i];

        w->zmq_socket = zmq_socket(g_zmq_context, ZMQ_REQ);

        /* Short timeout for the ping phase so we can retry quickly */
        int ping_timeout = PING_RETRY_SLEEP_MS;  /* ms */
        zmq_setsockopt(w->zmq_socket, ZMQ_RCVTIMEO, &ping_timeout, sizeof(ping_timeout));
        zmq_setsockopt(w->zmq_socket, ZMQ_SNDTIMEO, &ping_timeout, sizeof(ping_timeout));

        if (zmq_connect(w->zmq_socket, w->socket_addr) != 0) {
            fprintf(stderr, "Failed to connect to worker %d: %s\n",
                    i, zmq_strerror(errno));
            return -1;
        }

        int ready = 0;
        for (int attempt = 0; attempt < PING_MAX_RETRIES && !ready && g_running; attempt++) {

            /* Detect early worker death before wasting a ping timeout. */
            if (w->pid > 0) {
                int wstatus;
                pid_t dead = waitpid(w->pid, &wstatus, WNOHANG);
                if (dead > 0) {
                    fprintf(stderr,
                            "\nWorker %d (PID %d) exited prematurely (exit code %d).\n"
                            "Check that all Python dependencies are installed "
                            "(pydub, zmq, etc.).\n",
                            i, (int)w->pid,
                            WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1);
                    w->pid = 0;  /* already reaped — don't kill again in cleanup */
                    return -1;
                }
            }

            struct json_object *ping = json_object_new_object();
            json_object_object_add(ping, "command", json_object_new_string("ping"));
            const char *ping_str = json_object_to_json_string(ping);

            if (zmq_send(w->zmq_socket, ping_str, strlen(ping_str), 0) >= 0) {
                char buf[256];
                int rc = zmq_recv(w->zmq_socket, buf, sizeof(buf) - 1, 0);
                if (rc >= 0) {
                    ready = 1;
                }
            }
            json_object_put(ping);

            if (!ready && g_running) {
                /* ZMQ REQ socket enters an error state after a failed send/recv;
                 * we must close and re-open it to retry. */
                zmq_close(w->zmq_socket);
                w->zmq_socket = NULL;
                usleep(PING_RETRY_SLEEP_MS * 1000UL);

                w->zmq_socket = zmq_socket(g_zmq_context, ZMQ_REQ);
                zmq_setsockopt(w->zmq_socket, ZMQ_RCVTIMEO,
                               &ping_timeout, sizeof(ping_timeout));
                zmq_setsockopt(w->zmq_socket, ZMQ_SNDTIMEO,
                               &ping_timeout, sizeof(ping_timeout));
                zmq_connect(w->zmq_socket, w->socket_addr);
            }
        }

        if (!g_running) {
            fprintf(stderr, "\nInterrupted while waiting for workers.\n");
            return -1;
        }

        if (!ready) {
            fprintf(stderr, "Worker %d not responding after %d retries\n",
                    i, PING_MAX_RETRIES);
            return -1;
        }

        /* Switch to full-length timeout for actual inference */
        int inference_timeout = 300000;  /* 5 minutes */
        zmq_setsockopt(w->zmq_socket, ZMQ_RCVTIMEO,
                       &inference_timeout, sizeof(inference_timeout));
        zmq_setsockopt(w->zmq_socket, ZMQ_SNDTIMEO,
                       &inference_timeout, sizeof(inference_timeout));

        printf("Worker %d ready\n", i);
    }

    return 0;
}

/* -----------------------------------------------------------------------
 * Per-worker thread — dynamic work queue variant.
 *
 * OPTIMIZATION: original code divided chunks into static groups up-front and
 * each thread processed only its assigned range.  If chunk N took 3× longer
 * than average (longer text), its worker became the critical-path bottleneck
 * while all other workers sat idle.
 *
 * New: each thread pops the next available chunk from g_work_queue, processes
 * it, then pops the next one.  Fast workers absorb more work automatically.
 * --------------------------------------------------------------------- */
typedef struct {
    int     worker_id;
    ChunkList *chunks;
    char   *voice_path;
    char   *run_dir;
    int    *success_flags;
    double *durations;
    char  **error_msgs;
} WorkerThreadArgs;

void* worker_thread(void *arg) {
    WorkerThreadArgs *a = (WorkerThreadArgs *)arg;
    Worker *w = &g_workers[a->worker_id];

    int chunk_idx;
    while ((chunk_idx = work_queue_pop(&g_work_queue)) >= 0 && g_running) {

        /* Build output path */
        char chunk_path[2048];
        snprintf(chunk_path, sizeof(chunk_path), "%s/chunk_%d.mp3",
                 a->run_dir, chunk_idx);

        /* Build JSON request */
        struct json_object *req = json_object_new_object();
        json_object_object_add(req, "command",
                               json_object_new_string("process"));
        json_object_object_add(req, "text",
                               json_object_new_string(a->chunks->chunks[chunk_idx]));
        json_object_object_add(req, "voice_path",
                               json_object_new_string(a->voice_path));
        json_object_object_add(req, "output_path",
                               json_object_new_string(chunk_path));
        json_object_object_add(req, "chunk_index",
                               json_object_new_int(chunk_idx));

        const char *req_str = json_object_to_json_string(req);

        pthread_mutex_lock(&w->lock);
        int send_ok = (zmq_send(w->zmq_socket, req_str,
                                strlen(req_str), 0) >= 0);
        json_object_put(req);

        if (!send_ok) {
            pthread_mutex_unlock(&w->lock);
            snprintf(a->error_msgs[chunk_idx], 1024,
                     "Send failed: %s", zmq_strerror(errno));
            a->success_flags[chunk_idx] = 0;
            pthread_mutex_lock(&g_progress_mutex);
            g_completed_chunks++;
            pthread_mutex_unlock(&g_progress_mutex);
            display_progress_bar();
            continue;
        }

        char buf[65536];
        int rc = zmq_recv(w->zmq_socket, buf, sizeof(buf) - 1, 0);
        pthread_mutex_unlock(&w->lock);

        if (rc < 0) {
            snprintf(a->error_msgs[chunk_idx], 1024,
                     "Receive failed: %s", zmq_strerror(errno));
            a->success_flags[chunk_idx] = 0;
            pthread_mutex_lock(&g_progress_mutex);
            g_completed_chunks++;
            pthread_mutex_unlock(&g_progress_mutex);
            display_progress_bar();
            continue;
        }

        buf[rc] = '\0';

        struct json_object *resp = json_tokener_parse(buf);
        if (!resp) {
            snprintf(a->error_msgs[chunk_idx], 1024, "JSON parse failed");
            a->success_flags[chunk_idx] = 0;
            pthread_mutex_lock(&g_progress_mutex);
            g_completed_chunks++;
            pthread_mutex_unlock(&g_progress_mutex);
            display_progress_bar();
            continue;
        }

        struct json_object *status_obj;
        if (json_object_object_get_ex(resp, "status", &status_obj)) {
            const char *status = json_object_get_string(status_obj);

            if (strcmp(status, "success") == 0) {
                a->success_flags[chunk_idx] = 1;

                struct json_object *dur_obj;
                if (json_object_object_get_ex(resp, "duration_seconds", &dur_obj))
                    a->durations[chunk_idx] = json_object_get_double(dur_obj);

            } else {
                struct json_object *err_obj;
                if (json_object_object_get_ex(resp, "error", &err_obj))
                    strncpy(a->error_msgs[chunk_idx],
                            json_object_get_string(err_obj), 1023);
                a->success_flags[chunk_idx] = 0;
            }
        }

        json_object_put(resp);

        pthread_mutex_lock(&g_progress_mutex);
        g_completed_chunks++;
        pthread_mutex_unlock(&g_progress_mutex);
        display_progress_bar();
    }

    return NULL;
}

/* -----------------------------------------------------------------------
 * ffmpeg concatenation — fork+execvp instead of system()
 *
 * OPTIMIZATION: system(cmd) spawns /bin/sh which then spawns ffmpeg — two
 * processes for one job, plus shell quoting hazards.  execvp is direct.
 * --------------------------------------------------------------------- */
int concat_mp3s_ffmpeg(const char *list_path, const char *output_path) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork for ffmpeg");
        return -1;
    }
    if (pid == 0) {
        /* Child: exec ffmpeg directly */
        execlp("ffmpeg",
               "ffmpeg",
               "-f", "concat",
               "-safe", "0",
               "-i", list_path,
               "-c", "copy",
               "-y",
               output_path,
               NULL);
        perror("execlp ffmpeg");
        exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/* Check whether a binary exists on PATH (replaces system("which ...")) */
int binary_on_path(const char *name) {
    const char *path_env = getenv("PATH");
    if (!path_env) return 0;

    char *path_copy = strdup(path_env);
    char *dir = strtok(path_copy, ":");
    int found = 0;

    while (dir) {
        char candidate[4096];
        snprintf(candidate, sizeof(candidate), "%s/%s", dir, name);
        if (access(candidate, X_OK) == 0) {
            found = 1;
            break;
        }
        dir = strtok(NULL, ":");
    }

    free(path_copy);
    return found;
}

int main(int argc, char **argv) {
    Config cfg = {
        .text           = NULL,
        .voice_path     = NULL,
        .output_path    = NULL,
        .runs_dir       = "runs",
        .run_id         = NULL,
        .max_chunk_size = 500,
        .num_workers    = 4,
        .precision      = 0,    /* int8 default */
        .temperature    = 0.7f,
        .lsd_steps      = 10
    };

    static struct option long_options[] = {
        {"text_file",    required_argument, 0, 't'},
        {"speaker_wav",  required_argument, 0, 's'},
        {"workers",      required_argument, 0, 'w'},
        {"max_chunk_size", required_argument, 0, 'm'},
        {"output",       required_argument, 0, 'o'},
        {"runs_dir",     required_argument, 0, 'r'},
        {"run_id",       required_argument, 0, 'i'},
        {"precision",    required_argument, 0, 'p'},
        {"temperature",  required_argument, 0, 'T'},
        {"lsd_steps",    required_argument, 0, 'l'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv,
                              "t:s:w:m:o:r:i:p:T:l:h",
                              long_options, NULL)) != -1) {
        switch (opt) {
            case 't': cfg.text          = optarg; break;
            case 's': cfg.voice_path    = optarg; break;
            case 'w':
                cfg.num_workers = atoi(optarg);
                if (cfg.num_workers < 1 || cfg.num_workers > MAX_WORKERS) {
                    fprintf(stderr, "Workers must be 1-%d\n", MAX_WORKERS);
                    return 1;
                }
                break;
            case 'm': cfg.max_chunk_size = atoi(optarg); break;
            case 'o': cfg.output_path   = optarg; break;
            case 'r': cfg.runs_dir      = optarg; break;
            case 'i': cfg.run_id        = optarg; break;
            case 'p':
                cfg.precision = (strcmp(optarg, "fp32") == 0) ? 1 : 0; break;
            case 'T': cfg.temperature   = (float)atof(optarg); break;
            case 'l': cfg.lsd_steps     = atoi(optarg); break;
            case 'h':
            default:
                printf("Usage: %s [OPTIONS]\n", argv[0]);
                printf("  --text_file FILE        Input text file\n");
                printf("  --speaker_wav FILE      Reference voice WAV file\n");
                printf("  --workers N             Number of parallel workers (default: 4)\n");
                printf("  --max_chunk_size N      Max characters per chunk (default: 500)\n");
                printf("  --output FILE           Final combined output (optional)\n");
                printf("  --runs_dir DIR          Directory for run outputs (default: runs)\n");
                printf("  --run_id ID             Run identifier (default: auto timestamp)\n");
                printf("  --precision int8|fp32   Model precision (default: int8)\n");
                printf("  --temperature FLOAT     Generation temperature (default: 0.7)\n");
                printf("  --lsd_steps N           LSD steps for quality (default: 10)\n");
                return 0;
        }
    }

    if (!cfg.text || !cfg.voice_path) {
        fprintf(stderr, "Error: --text_file and --speaker_wav are required\n");
        return 1;
    }

    /* Generate run_id if not provided */
    char auto_run_id[32];
    if (!cfg.run_id) {
        generate_run_id(auto_run_id, sizeof(auto_run_id));
        cfg.run_id = auto_run_id;
    }
    strncpy(g_run_id_global, cfg.run_id, sizeof(g_run_id_global) - 1);

    char run_dir[2048];
    snprintf(run_dir, sizeof(run_dir), "%s/%s", cfg.runs_dir, cfg.run_id);

    mkdir_p(run_dir);
    printf("Run directory: %s\n", run_dir);

    /* Read input file */
    FILE *f = fopen(cfg.text, "rb");
    if (!f) {
        perror("Failed to open text file");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long text_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *text_content = malloc(text_size + 1);
    if (fread(text_content, 1, text_size, f) != (size_t)text_size)
        perror("fread text");
    text_content[text_size] = '\0';
    fclose(f);

    printf("Loaded text file: %ld bytes\n", text_size);
    printf("Speaker: %s\n", cfg.voice_path);
    printf("Workers: %d\n", cfg.num_workers);
    printf("Max chunk size: %d\n", cfg.max_chunk_size);
    printf("Run ID: %s\n", cfg.run_id);
    printf("Output format: MP3 @ 192kbps\n");

    /* OPTIMIZATION: use binary_on_path() instead of system("which ffmpeg ...") */
    int ffmpeg_available = binary_on_path("ffmpeg");
    if (!ffmpeg_available && cfg.output_path)
        fprintf(stderr, "Warning: ffmpeg not found — final concatenation will fail\n");

    /* Split into chunks */
    ChunkList *chunks = split_text(text_content, cfg.max_chunk_size);
    printf("Split into %d chunks (all will be MP3 files)\n", chunks->count);

    if (chunks->count == 0) {
        fprintf(stderr, "No chunks to process\n");
        free(text_content);
        free_chunks(chunks);
        return 1;
    }

    if (cfg.num_workers > chunks->count) {
        printf("Note: Reducing workers from %d to %d (one per chunk)\n",
               cfg.num_workers, chunks->count);
        cfg.num_workers = chunks->count;
    }

    /* Start workers */
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);
    atexit(cleanup);

    int allocated = allocate_cores(cfg.run_id, cfg.num_workers);
    if (allocated <= 0) {
        fprintf(stderr, "Failed to allocate CPU cores\n");
        return 1;
    }
    cfg.num_workers = allocated;

    printf("Allocated cores: ");
    for (int i = 0; i < allocated; i++)
        printf("%d ", g_allocated_cores[i]);
    printf("\n");

    if (start_workers(&cfg) < 0) {
        fprintf(stderr, "Failed to start workers\n");
        return 1;
    }

    /* Allocate result tracking */
    int    *success_flags = calloc(chunks->count, sizeof(int));
    double *durations     = calloc(chunks->count, sizeof(double));
    char  **error_msgs    = calloc(chunks->count, sizeof(char*));
    for (int i = 0; i < chunks->count; i++)
        error_msgs[i] = calloc(1024, sizeof(char));

    /* Initialize dynamic work queue */
    work_queue_init(&g_work_queue, chunks->count);

    /* Initialize progress */
    g_completed_chunks = 0;
    g_total_chunks     = chunks->count;
    g_start_time       = time(NULL);

    printf("Processing %d chunks across %d workers (dynamic queue)...\n",
           chunks->count, cfg.num_workers);
    display_progress_bar();
    printf("\n");

    /* Launch one thread per worker — each pulls from the shared queue */
    pthread_t        *threads = calloc(cfg.num_workers, sizeof(pthread_t));
    WorkerThreadArgs *targs   = calloc(cfg.num_workers, sizeof(WorkerThreadArgs));

    for (int i = 0; i < cfg.num_workers && g_running; i++) {
        targs[i].worker_id    = i;
        targs[i].chunks       = chunks;
        targs[i].voice_path   = cfg.voice_path;
        targs[i].run_dir      = run_dir;
        targs[i].success_flags = success_flags;
        targs[i].durations    = durations;
        targs[i].error_msgs   = error_msgs;

        pthread_create(&threads[i], NULL, worker_thread, &targs[i]);

        /* OPTIMIZATION: removed usleep(100000) stagger — threads start
         * immediately and pull from the queue; no reason to delay. */
    }

    for (int i = 0; i < cfg.num_workers; i++)
        pthread_join(threads[i], NULL);

    work_queue_destroy(&g_work_queue);

    printf("\n=== Processing Complete ===\n");

    /* Calculate results */
    int    success_count  = 0;
    double total_duration = 0.0;

    for (int i = 0; i < chunks->count; i++) {
        if (success_flags[i]) {
            success_count++;
            total_duration += durations[i];
        } else {
            fprintf(stderr, "Chunk %d failed: %s\n", i, error_msgs[i]);
        }
    }

    /* Write manifest */
    char manifest_path[2048];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", run_dir);
    FILE *manifest = fopen(manifest_path, "w");

    if (manifest) {
        fprintf(manifest, "{\n");
        fprintf(manifest, "  \"run_id\": \"%s\",\n", cfg.run_id);
        fprintf(manifest, "  \"total_chunks\": %d,\n", chunks->count);
        fprintf(manifest, "  \"successful_chunks\": %d,\n", success_count);
        fprintf(manifest, "  \"workers_used\": %d,\n", cfg.num_workers);
        fprintf(manifest, "  \"chunks\": [\n");

        for (int i = 0; i < chunks->count; i++) {
            if (success_flags[i]) {
                fprintf(manifest, "    {\n");
                fprintf(manifest, "      \"index\": %d,\n", i);
                fprintf(manifest, "      \"file\": \"chunk_%d.mp3\",\n", i);
                fprintf(manifest, "      \"duration_seconds\": %.3f,\n", durations[i]);
                fprintf(manifest, "      \"text_preview\": \"");
                int count = 0;
                for (char *p = chunks->chunks[i]; *p && count < 50; p++, count++) {
                    if (*p == '"' || *p == '\\') fputc('\\', manifest);
                    fputc(*p, manifest);
                }
                if (strlen(chunks->chunks[i]) > 50) fprintf(manifest, "...");
                fprintf(manifest, "\"\n");
                fprintf(manifest, "    }");
            } else {
                fprintf(manifest, "    {\n");
                fprintf(manifest, "      \"index\": %d,\n", i);
                fprintf(manifest, "      \"file\": null,\n");
                fprintf(manifest, "      \"error\": \"");
                for (char *p = error_msgs[i]; *p; p++) {
                    if (*p == '"' || *p == '\\') fputc('\\', manifest);
                    fputc(*p, manifest);
                }
                fprintf(manifest, "\"\n");
                fprintf(manifest, "    }");
            }
            if (i < chunks->count - 1) fprintf(manifest, ",");
            fprintf(manifest, "\n");
        }

        fprintf(manifest, "  ],\n");
        fprintf(manifest, "  \"total_duration_seconds\": %.3f\n", total_duration);
        fprintf(manifest, "}\n");
        fclose(manifest);
        printf("Manifest saved to: %s\n", manifest_path);
    }

    printf("\nResults: %d/%d chunks successful\n", success_count, chunks->count);
    printf("Total audio duration: %.2f seconds (%.2f minutes)\n",
           total_duration, total_duration / 60.0);
    printf("Output directory: %s\n", run_dir);

    /* Optionally combine chunks — use fork+execvp instead of system() */
    if (cfg.output_path && success_count > 0) {
        char final_output[2048];
        if (!strstr(cfg.output_path, ".mp3"))
            snprintf(final_output, sizeof(final_output), "%s.mp3", cfg.output_path);
        else
            snprintf(final_output, sizeof(final_output), "%s", cfg.output_path);

        printf("\nCombining %d MP3 chunks into: %s\n", success_count, final_output);

        char list_path[2048];
        snprintf(list_path, sizeof(list_path), "%s/concat_list.txt", run_dir);
        FILE *list_file = fopen(list_path, "w");

        if (list_file) {
            for (int i = 0; i < chunks->count; i++)
                if (success_flags[i])
                    fprintf(list_file, "file 'chunk_%d.mp3'\n", i);
            fclose(list_file);

            /* OPTIMIZATION: fork+execvp — no shell process spawned */
            int ret = concat_mp3s_ffmpeg(list_path, final_output);
            if (ret == 0)
                printf("✓ Final MP3 output saved to: %s\n", final_output);
            else
                fprintf(stderr, "Warning: ffmpeg combine failed (exit code %d)\n", ret);
        }
    }

    /* Cleanup */
    for (int i = 0; i < chunks->count; i++)
        free(error_msgs[i]);
    free(error_msgs);
    free(success_flags);
    free(durations);
    free(targs);
    free(threads);
    free(text_content);
    free_chunks(chunks);

    return (success_count == chunks->count) ? 0 : 1;
}