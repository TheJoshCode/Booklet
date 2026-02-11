import threading

class TTSState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.current_idx = 0
        self.total_chunks = 0
        self.eta_seconds = 0

        self.pause_event.set()  # not paused initially

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def stop(self):
        self.stop_event.set()

    def reset(self, total=0):
        self.current_idx = 0
        self.total_chunks = total
        self.stop_event.clear()
        self.pause_event.set()
        self.eta_seconds = 0

state = TTSState()
