import threading

class GlobalState:
    def __init__(self):
        self._train_stop = False
        self._infer_stop = False
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self._train_stop = self._infer_stop = False

    def stop_train(self):
        with self.lock:
            self._train_stop = True

    def stop_infer(self):
        with self.lock:
            self._infer_stop = True

    @property
    def train_stop(self):
        with self.lock:
            return self._train_stop

    @property
    def infer_stop(self):
        with self.lock:
            return self._infer_stop


global_state = GlobalState()