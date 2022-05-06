from time import time
import numpy as np
import torch


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    """A flexible Timer class. (credit to mmcv)

    Examples:
        >>> import time
        >>> with Timer():
        >>>     time.sleep(1)
        1.000
        >>> with Timer(print_tmpl='it takes {:.1f} seconds'):
        >>>     time.sleep(1)
        it takes 1.0 seconds
        >>> timer = Timer()
        >>> time.sleep(0.5)
        >>> print(timer.since_start())
        0.500
        >>> time.sleep(0.5)
        >>> print(timer.since_last_check())
        0.500
        >>> print(timer.since_start())
        1.000
    """

    def __init__(self, print_tmpl=None, start=True):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        """Total time since the timer is started.

        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur


class IterTimer:
    """credit to Hansheng Chen."""

    def __init__(self, name='time', sync=True, enabled=True):
        self.name = name
        self.times = []
        self.timer = Timer(start=False)
        self.sync = sync
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer.start()
        return self

    def __exit__(self, type, value, traceback):
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer_record()
        self.timer._is_running = False

    def timer_start(self):
        self.timer.start()

    def timer_record(self):
        self.times.append(self.timer.since_last_check())

    def print_time(self):
        if not self.enabled:
            return
        print('Average {} = {:.4f}'.format(self.name, np.average(self.times)))


class IterTimers(dict):
    def __init__(self, *args, **kwargs):
        super(IterTimers, self).__init__(*args, **kwargs)

    def disable_all(self):
        for timer in self.values():
            timer.enabled = False

    def enable_all(self):
        for timer in self.values():
            timer.enabled = True

    def add_timer(self, name='time', sync=True, enabled=False):
        self[name] = IterTimer(
            name, sync=sync, enabled=enabled)