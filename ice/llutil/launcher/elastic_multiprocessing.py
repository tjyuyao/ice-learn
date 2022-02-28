import logging
import os
import threading
import time
from concurrent.futures._base import Future
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
import traceback
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union
from ice.llutil.logger import get_logger
from ice.llutil.launcher import events
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.api import (  # noqa: F401
    ProcessFailure,
    RunProcsResult,
    Std,
    SignalException,
    SubprocessContext,
    _validate_full_rank,
    to_map,
)

log = get_logger()


def tail_logfile(
    header: str, file: str, dst: TextIO, finished: Event, interval_sec: float,
    lock: threading.Lock, progbar_events: Dict[str, threading.Event]
):
    
    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)

    with open(file, "r") as fp:
        while True:
            line = fp.readline()[:-1]

            if line:
                    lock.acquire(True)
                    progbar_events["trigger_clear"].set()
                    progbar_events["cleared"].wait()
                    dst.write(f"\r{line}\033[K\n")
                    progbar_events["trigger_clear"].clear()
                    lock.release()
                    
            else:  # reached EOF
                if finished.is_set():
                    # log line producer is finished
                    break
                else:
                    # log line producer is still going
                    # wait for a bit before looping again
                    time.sleep(interval_sec)

        


class TailLog:
    """
    Tails the given log files. The log files do not have to exist when the
    ``start()`` method is called. The tail-er will gracefully wait until the
    log files are created by the producer and will tail the contents of the
    log files until the ``stop()`` method is called.

    .. warning:: ``TailLog`` will wait indefinitely for the log file to be created!

    Each log file's line will be suffixed with a header of the form: ``[{name}{idx}]:``,
    where the ``name`` is user-provided and ``idx`` is the index of the log file
    in the ``log_files`` mapping.

    Usage:

    ::

     log_files = {0: "/tmp/0_stdout.log", 1: "/tmp/1_stdout.log"}
     tailer = TailLog("trainer", log_files, sys.stdout).start()
     # actually run the trainers to produce 0_stdout.log and 1_stdout.log
     run_trainers()
     tailer.stop()

     # once run_trainers() start writing the ##_stdout.log files
     # the tailer will print to sys.stdout:
     # >>> [trainer0]:log_line1
     # >>> [trainer1]:log_line1
     # >>> [trainer0]:log_line2
     # >>> [trainer0]:log_line3
     # >>> [trainer1]:log_line2

    .. note:: Due to buffering log lines between files may not necessarily
              be printed out in order. You should configure your application's
              logger to suffix each log line with a proper timestamp.

    """

    def __init__(
        self,
        name: str,
        log_files: Dict[int, str],
        dst: TextIO,
        interval_sec: float = 0.1,
        progbar_events = None,
    ):
        n = len(log_files)
        self._threadpool = None
        if n > 0:
            self._threadpool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix=f"{self.__class__.__qualname__}_{name}",
            )

        self._name = name
        self._dst = dst
        self._log_files = log_files
        self._finished_events: Dict[int, Event] = {
            local_rank: Event() for local_rank in log_files.keys()
        }
        self._progbar_lock = threading.Lock()
        self._progbar_events = progbar_events
        self._futs: List[Future] = []
        self._interval_sec = interval_sec
        self._stopped = False

    def start(self) -> "TailLog":
        if not self._threadpool:
            return self

        for local_rank, file in self._log_files.items():
            self._futs.append(
                self._threadpool.submit(
                    tail_logfile,
                    # header=f"[{self._name}{local_rank}]:",
                    header="",
                    file=file,
                    dst=self._dst,
                    finished=self._finished_events[local_rank],
                    interval_sec=self._interval_sec,
                    lock=self._progbar_lock,
                    progbar_events=self._progbar_events,
                )
            )
        return self

    def stop(self) -> None:
        for finished in self._finished_events.values():
            finished.set()

        for local_rank, f in enumerate(self._futs):
            try:
                f.result()
            except Exception as e:
                log.error(
                    f"error in log tailor for {self._name}{local_rank}."
                    f" {e.__class__.__qualname__}: {e}",
                )
                traceback.print_tb(e.__traceback__)

        if self._threadpool:
            self._threadpool.shutdown(wait=True)

        self._stopped = True

    def stopped(self) -> bool:
        return self._stopped

import abc
import signal
import sys

from torch.distributed.elastic.multiprocessing.api import (
    _validate_full_rank,
    _terminate_process_handler,
    _get_default_signal,
    _get_kill_signal,
    _wrap,
    IS_WINDOWS,
    RunProcsResult,
)

class PContext(abc.ABC):
    """
    The base class that standardizes operations over a set of processes
    that are launched via different mechanisms. The name ``PContext``
    is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

    .. warning:: stdouts and stderrs should ALWAYS be a superset of
                 tee_stdouts and tee_stderrs (respectively) this is b/c
                 tee is implemented as a redirect + tail -f <stdout/stderr.log>
    """

    def __init__(
        self,
        name: str,
        entrypoint: Union[Callable, str],
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        stdouts: Dict[int, str],
        stderrs: Dict[int, str],
        tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str],
        error_files: Dict[int, str],
        progbar_events: Dict[str, threading.Event],
    ):
        self.name = name
        # validate that all mappings have the same number of keys and
        # all local ranks are accounted for
        nprocs = len(args)
        _validate_full_rank(stdouts, nprocs, "stdouts")
        _validate_full_rank(stderrs, nprocs, "stderrs")

        self.entrypoint = entrypoint
        self.args = args
        self.envs = envs
        self.stdouts = stdouts
        self.stderrs = stderrs
        self.error_files = error_files
        self.nprocs = nprocs
        
        self._stdout_tail = TailLog(name, tee_stdouts, sys.stdout, progbar_events=progbar_events)
        self._stderr_tail = TailLog(name, tee_stderrs, sys.stderr, progbar_events=progbar_events)

    def start(self) -> None:
        """
        Start processes using parameters defined in the constructor.
        """
        signal.signal(signal.SIGTERM, _terminate_process_handler)
        signal.signal(signal.SIGINT, _terminate_process_handler)
        if not IS_WINDOWS:
            signal.signal(signal.SIGHUP, _terminate_process_handler)
            signal.signal(signal.SIGQUIT, _terminate_process_handler)
        self._start()
        self._stdout_tail.start()
        self._stderr_tail.start()

    @abc.abstractmethod
    def _start(self) -> None:
        """
        Start processes using strategy defined in a particular context.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _poll(self) -> Optional[RunProcsResult]:
        """
        Polls the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        """
        raise NotImplementedError()

    def wait(self, timeout: float = -1, period: float = 1) -> Optional[RunProcsResult]:
        """
        Waits for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        ..note: Multiprocesing library registers SIGTERM and SIGINT signal handlers that raise
                ``SignalException`` when the signals received. It is up to the consumer of the code
                to properly handle the exception. It is important not to swallow the exception otherwise
                the process would not terminate. Example of the typical workflow can be:

        .. code-block:: python
            pc = start_processes(...)
            try:
                pc.wait(1)
                .. do some other work
            except SignalException as e:
                pc.shutdown(e.sigval, timeout=30)

        If SIGTERM or SIGINT occurs, the code above will try to shutdown child processes by propagating
        received signal. If child processes will not terminate in the timeout time, the process will send
        the SIGKILL.
        """

        if timeout == 0:
            return self._poll()

        if timeout < 0:
            timeout = sys.maxsize

        expiry = time.time() + timeout
        while time.time() < expiry:
            pr = self._poll()
            if pr:
                return pr
            time.sleep(period)

        return None

    @abc.abstractmethod
    def pids(self) -> Dict[int, int]:
        """
        Returns pids of processes mapped by their respective local_ranks
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
        raise NotImplementedError()

    def close(
        self, death_sig: Optional[signal.Signals] = None, timeout: int = 30
    ) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate porcesses.
            timeout: Time to wait for processes to finish, if process is
                still alive after this time, it will be terminated via SIGKILL.
        """
        if not death_sig:
            death_sig = _get_default_signal()
        self._close(death_sig=death_sig, timeout=timeout)
        if self._stdout_tail:
            self._stdout_tail.stop()
        if self._stderr_tail:
            self._stderr_tail.stop()


class MultiprocessContext(PContext):
    """
    ``PContext`` holding worker processes invoked as a function.
    """

    def __init__(
        self,
        name: str,
        entrypoint: Callable,
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        stdouts: Dict[int, str],
        stderrs: Dict[int, str],
        tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str],
        error_files: Dict[int, str],
        start_method: str,
        progbar_events,
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            stdouts,
            stderrs,
            tee_stdouts,
            tee_stderrs,
            error_files,
            progbar_events,
        )

        self.start_method = start_method
        # each ret_val queue will always contain a single element.
        self._ret_vals = {
            local_rank: mp.get_context(self.start_method).SimpleQueue()
            for local_rank in range(self.nprocs)
        }

        # see comments in ``join()`` for what this is
        self._return_values: Dict[int, Any] = {}
        self._pc: Optional[mp.ProcessContext] = None
        # Note: set method should ONLY be invoked for the use case when all processes finished
        # successfully. If any process died on event.wait() calling set() method will deadlock.
        self._worker_finished_event = mp.get_context(self.start_method).Event()

    def _start(self):
        if self._pc:
            raise ValueError(
                "The process context already initialized."
                " Most likely the start method got called twice."
            )
        self._pc = mp.start_processes(
            fn=_wrap,
            args=(
                self.entrypoint,
                self.args,
                self.envs,
                self.stdouts,
                self.stderrs,
                self._ret_vals,
                self._worker_finished_event,
            ),
            nprocs=self.nprocs,
            join=False,
            daemon=False,
            start_method=self.start_method,
        )

    def _is_done(self) -> bool:
        return len(self._return_values) == self.nprocs

    def _poll(self) -> Optional[RunProcsResult]:
        assert self._pc is not None  # assertion for mypy type checker

        try:
            # torch.mp.ProcessContext Throws an Exception if some/all of
            # worker processes failed
            # timeout < 0 checks worker status and return immediately
            # Join will never return success since we use synchronize.Event to wait
            # for all processes to finish.
            self._pc.join(-1)

            # IMPORTANT: we use multiprocessing.Queue to carry worker return values
            # back to the parent, the worker process will wait before terminating
            # until all the buffered items are fed by the feeder thread to the underlying
            # pipe. Hence to prevent deadlocks on large return values,
            # we opportunistically try queue.get on each join call
            # See: https://docs.python.org/2/library/multiprocessing.html#all-platforms
            for local_rank in range(0, self.nprocs):
                return_queue = self._ret_vals[local_rank]
                if not return_queue.empty():
                    # save the return values temporarily into a member var
                    self._return_values[local_rank] = return_queue.get()

            if self._is_done():
                # we should ALWAYS have ALL the return values when all the processes are done
                self._worker_finished_event.set()
                # Wait untill all processes are finished. At this point workers finished executing
                # user function
                self._pc.join()
                _validate_full_rank(
                    self._return_values, self.nprocs, "return_value queue"
                )
                self.close()
                return RunProcsResult(
                    return_values=self._return_values,
                    stdouts=self.stdouts,
                    stderrs=self.stderrs,
                )
            else:
                return None
        except (mp.ProcessRaisedException, mp.ProcessExitedException) as e:
            failed_local_rank = e.error_index

            # entrypoint for MultiprocessContext will always be a Callable
            fn_name = self.entrypoint.__qualname__  # type: ignore[union-attr]
            failed_proc = self._pc.processes[failed_local_rank]
            error_filepath = self.error_files[failed_local_rank]

            log.error(
                f"failed (exitcode: {failed_proc.exitcode})"
                f" local_rank: {failed_local_rank} (pid: {e.pid})"
                f" of fn: {fn_name} (start_method: {self.start_method})",
                exc_info=True,
            )

            self.close()
            return RunProcsResult(
                failures={
                    failed_local_rank: ProcessFailure(
                        local_rank=failed_local_rank,
                        pid=e.pid,
                        exitcode=failed_proc.exitcode,
                        error_file=error_filepath,
                    )
                },
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )

    def pids(self) -> Dict[int, int]:
        assert self._pc is not None  # assertion for mypy type checking
        return {local_rank: pid for local_rank, pid in enumerate(self._pc.pids())}

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not self._pc:
            return
        for proc in self._pc.processes:
            if proc.is_alive():
                log.warning(f"Closing process {proc.pid} via signal {death_sig.name}")
                try:
                    os.kill(proc.pid, death_sig)
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be rasied, it is safe to ignore it.
                    pass
        end = time.monotonic() + timeout
        for proc in self._pc.processes:
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            proc.join(time_to_wait)
        for proc in self._pc.processes:
            if proc.is_alive():
                log.warning(
                    f"Unable to shutdown process {proc.pid} via {death_sig}, forcefully exitting via {_get_kill_signal()}"
                )
                try:
                    os.kill(proc.pid, _get_kill_signal())
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be rasied, it is safe to ignore it.
                    pass
            proc.join()


def start_processes(
    name: str,
    entrypoint: Union[Callable, str],
    args: Dict[int, Tuple],
    envs: Dict[int, Dict[str, str]],
    log_dir: str,
    start_method: str = "spawn",
    redirects: Union[Std, Dict[int, Std]] = Std.NONE,
    tee: Union[Std, Dict[int, Std]] = Std.NONE,
    progbar_events: Dict[str, threading.Event] = None,
) -> PContext:
    """
    Starts ``n`` copies of ``entrypoint`` processes with the provided options.
    ``entrypoint`` is either a ``Callable`` (function) or a ``str`` (binary).
    The number of copies is determined by the number of entries for ``args`` and
    ``envs`` arguments, which need to have the same key set.

    ``args`` and ``env`` parameters are the arguments and environment variables
    to pass down to the entrypoint mapped by the replica index (local rank).
    All local ranks must be accounted for.
    That is, the keyset should be ``{0,1,...,(nprocs-1)}``.

    .. note:: When the ``entrypoint`` is a binary (``str``), ``args`` can only be strings.
              If any other type is given, then it is casted to a string representation
              (e.g. ``str(arg1)``). Furthermore, a binary failure will only write
              an ``error.json`` error file if the main function is annotated with
              ``torch.distributed.elastic.multiprocessing.errors.record``. For function launches,
              this is done by default and there is no need to manually annotate
              with the ``@record`` annotation.

    ``redirects`` and ``tees`` are bitmasks specifying which std stream(s) to redirect
    to a log file in the ``log_dir``. Valid mask values are defined in ``Std``.
    To redirect/tee only certain local ranks, pass ``redirects`` as a map with the key as
    the local rank to specify the redirect behavior for.
    Any missing local ranks will default to ``Std.NONE``.

    ``tee`` acts like the unix "tee" command in that it redirects + prints to console.
    To avoid worker stdout/stderr from printing to console, use the ``redirects`` parameter.

    For each process, the ``log_dir`` will contain:

    #. ``{local_rank}/error.json``: if the process failed, a file with the error info
    #. ``{local_rank}/stdout.json``: if ``redirect & STDOUT == STDOUT``
    #. ``{local_rank}/stderr.json``: if ``redirect & STDERR == STDERR``

    .. note:: It is expected that the ``log_dir`` exists, is empty, and is a directory.

    Example:

    ::

     log_dir = "/tmp/test"

     # ok; two copies of foo: foo("bar0"), foo("bar1")
     start_processes(
        name="trainer",
        entrypoint=foo,
        args:{0:("bar0",), 1:("bar1",),
        envs:{0:{}, 1:{}},
        log_dir=log_dir
     )

     # invalid; envs missing for local rank 1
     start_processes(
        name="trainer",
        entrypoint=foo,
        args:{0:("bar0",), 1:("bar1",),
        envs:{0:{}},
        log_dir=log_dir
     )

     # ok; two copies of /usr/bin/touch: touch file1, touch file2
     start_processes(
        name="trainer",
        entrypoint="/usr/bin/touch",
        args:{0:("file1",), 1:("file2",),
        envs:{0:{}, 1:{}},
        log_dir=log_dir
      )

     # caution; arguments casted to string, runs:
     # echo "1" "2" "3" and echo "[1, 2, 3]"
     start_processes(
        name="trainer",
        entrypoint="/usr/bin/echo",
        args:{0:(1,2,3), 1:([1,2,3],),
        envs:{0:{}, 1:{}},
        log_dir=log_dir
      )

    Args:
        name: a human readable short name that describes what the processes are
              (used as header when tee'ing stdout/stderr outputs)
        entrypoint: either a ``Callable`` (function) or ``cmd`` (binary)
        args: arguments to each replica
        envs: env vars to each replica
        log_dir: directory used to write log files
        nprocs: number of copies to create (one on each process)
        start_method: multiprocessing start method (spawn, fork, forkserver)
                      ignored for binaries
        redirects: which std streams to redirect to a log file
        tees: which std streams to redirect + print to console

    """

    # listdir raises FileNotFound or NotADirectoryError so no need to check manually
    if os.listdir(log_dir):
        raise RuntimeError(
            f"log_dir: {log_dir} is not empty, please provide an empty log_dir"
        )

    nprocs = len(args)
    _validate_full_rank(args, nprocs, "args")
    _validate_full_rank(envs, nprocs, "envs")

    # create subdirs for each local rank in the logs_dir
    # logs_dir
    #       |- 0
    #          |- error.json
    #          |- stdout.log
    #          |- stderr.log
    #       |- ...
    #       |- (nprocs-1)
    redirs = to_map(redirects, nprocs)
    ts = to_map(tee, nprocs)

    # to tee stdout/stderr we first redirect into a file
    # then tail -f stdout.log/stderr.log so add tee settings to redirects
    for local_rank, tee_std in ts.items():
        redirect_std = redirs[local_rank]
        redirs[local_rank] = redirect_std | tee_std

    stdouts = {local_rank: "" for local_rank in range(nprocs)}
    stderrs = {local_rank: "" for local_rank in range(nprocs)}
    tee_stdouts: Dict[int, str] = {}
    tee_stderrs: Dict[int, str] = {}
    error_files = {}

    for local_rank in range(nprocs):
        clogdir = os.path.join(log_dir, str(local_rank))
        os.mkdir(clogdir)

        rd = redirs[local_rank]
        if (rd & Std.OUT) == Std.OUT:
            stdouts[local_rank] = os.path.join(clogdir, "stdout.log")
        if (rd & Std.ERR) == Std.ERR:
            stderrs[local_rank] = os.path.join(clogdir, "stderr.log")

        t = ts[local_rank]
        if t & Std.OUT == Std.OUT:
            tee_stdouts[local_rank] = stdouts[local_rank]
        if t & Std.ERR == Std.ERR:
            tee_stderrs[local_rank] = stderrs[local_rank]

        error_file = os.path.join(clogdir, "error.json")
        error_files[local_rank] = error_file
        log.info(f"Setting worker{local_rank} reply file to: {error_file}")
        envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

    context: PContext
    context = MultiprocessContext(
        name=name,
        entrypoint=entrypoint,
        args=args,
        envs=envs,
        stdouts=stdouts,
        stderrs=stderrs,
        tee_stdouts=tee_stdouts,
        tee_stderrs=tee_stderrs,
        error_files=error_files,
        start_method=start_method,
        progbar_events=progbar_events,
    )

    try:
        context.start()
        return context
    except Exception:
        context.close()
        raise
