import time


class Timer:
    """It is a simple wall-clock timer for measuring inference latency."""

    def __init__(self):
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Timer.stop_ms() called before Timer.start()")
        dt = time.perf_counter() - self._t0
        self._t0 = None
        return dt * 1000.0
