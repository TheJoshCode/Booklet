CC = gcc
CFLAGS = -Wall -Wextra -O2 -pthread
LDFLAGS = -lzmq -ljson-c -lpthread

TARGET = tts_cli
SRC = tts_cli.c

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -f /tmp/tts_worker_*.sock

install:
	cp $(TARGET) /usr/local/bin/
	cp worker_instance.py /usr/local/bin/
	chmod +x /usr/local/bin/worker_instance.py

# Run example
test: $(TARGET)
	./$(TARGET) --text_file sample.txt --speaker_wav reference.wav --workers 2 --max_chunk_size 200