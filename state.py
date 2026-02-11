import threading
import time


class State:
    def __init__(self):
        self.lock = threading.Lock()

        # Control flags
        self._paused = False
        self._stopped = False
        self._running = False

        # Progress tracking
        self.current_idx = 0
        self.total_chunks = 0
        self.failed_chunks = 0
        self.eta_seconds = 0

        self.start_time = None

    # =====================================================
    # Control Methods
    # =====================================================

    def start(self, total_chunks: int):
        with self.lock:
            self._paused = False
            self._stopped = False
            self._running = True
            self.current_idx = 0
            self.failed_chunks = 0
            self.total_chunks = total_chunks
            self.eta_seconds = 0
            self.start_time = time.time()

    def pause(self):
        with self.lock:
            self._paused = True

    def resume(self):
        with self.lock:
            self._paused = False

    def stop(self):
        with self.lock:
            self._stopped = True
            self._paused = False
            self._running = False

    def finish(self):
        with self.lock:
            self._running = False
            self.eta_seconds = 0

    def reset(self):
        with self.lock:
            self._paused = False
            self._stopped = False
            self._running = False
            self.current_idx = 0
            self.total_chunks = 0
            self.failed_chunks = 0
            self.eta_seconds = 0
            self.start_time = None

    # =====================================================
    # Safe Status Getters
    # =====================================================

    def is_paused(self):
        with self.lock:
            return self._paused

    def is_stopped(self):
        with self.lock:
            return self._stopped

    def is_running(self):
        with self.lock:
            return self._running

    # =====================================================
    # Progress Updates
    # =====================================================

    def increment_completed(self):
        with self.lock:
            self.current_idx += 1
            self._update_eta()

    def increment_failed(self):
        with self.lock:
            self.failed_chunks += 1
            self._update_eta()

    def _update_eta(self):
        if self.start_time and self.current_idx > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / self.current_idx
            # FIXED: Account for failed chunks in ETA
            remaining = self.total_chunks - self.current_idx - self.failed_chunks
            self.eta_seconds = int(avg_time * max(0, remaining))
        else:
            self.eta_seconds = 0


# Global state instance
state = State()