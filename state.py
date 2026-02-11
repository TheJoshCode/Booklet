import threading

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.paused = False
        self.stopped = False
        self.current_idx = 0
        self.total_chunks = 0
        self.eta_seconds = 0

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stopped = True
        self.paused = False

    def reset(self):
        with self.lock:
            self.paused = False
            self.stopped = False
            self.current_idx = 0
            self.total_chunks = 0
            self.eta_seconds = 0

# Global state instance
state = State()