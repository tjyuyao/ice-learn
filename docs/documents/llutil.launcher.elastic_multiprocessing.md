<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.launcher.elastic_multiprocessing`






**Global Variables**
---------------
- **IS_WINDOWS**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tail_logfile`

```python
tail_logfile(
    header: str,
    file: str,
    dst: <class 'TextIO'>,
    finished: Event,
    interval_sec: float,
    lock: <built-in function allocate_lock>,
    progbar_events: Dict[str, Event]
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `start_processes`

```python
start_processes(
    name: str,
    entrypoint: Union[Callable, str],
    args: Dict[int, Tuple],
    envs: Dict[int, Dict[str, str]],
    log_dir: str,
    start_method: str = 'spawn',
    redirects: Union[Std, Dict[int, Std]] = <Std.NONE: 0>,
    tee: Union[Std, Dict[int, Std]] = <Std.NONE: 0>,
    progbar_events: Dict[str, Event] = None
) → PContext
```

Starts `n` copies of `entrypoint` processes with the provided options.
`entrypoint` is either a `Callable` (function) or a `str` (binary).
The number of copies is determined by the number of entries for `args` and
`envs` arguments, which need to have the same key set.


`args` and `env` parameters are the arguments and environment variables
to pass down to the entrypoint mapped by the replica index (local rank).
All local ranks must be accounted for.
That is, the keyset should be ``{0,1,...,(nprocs-1)}``.


.. note:: When the `entrypoint` is a binary (`str`), `args` can only be strings.
 If any other type is given, then it is casted to a string representation
 (e.g. `str(arg1)`). Furthermore, a binary failure will only write
 an `error.json` error file if the main function is annotated with
 `torch.distributed.elastic.multiprocessing.errors.record`. For function launches,
 this is done by default and there is no need to manually annotate
 with the ``@record`` annotation.


`redirects` and `tees` are bitmasks specifying which std stream(s) to redirect
to a log file in the `log_dir`. Valid mask values are defined in `Std`.
To redirect/tee only certain local ranks, pass `redirects` as a map with the key as
the local rank to specify the redirect behavior for.
Any missing local ranks will default to `Std.NONE`.


`tee` acts like the unix "tee" command in that it redirects + prints to console.
To avoid worker stdout/stderr from printing to console, use the `redirects` parameter.


For each process, the `log_dir` will contain:


#. ``{local_rank}/error.json``: if the process failed, a file with the error info
#. ``{local_rank}/stdout.json``: if ``redirect & STDOUT == STDOUT``
#. ``{local_rank}/stderr.json``: if ``redirect & STDERR == STDERR``


.. note:: It is expected that the `log_dir` exists, is empty, and is a directory.




**Example:**



:
```


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


```


**Args:**


 - <b>`name`</b>:  a human readable short name that describes what the processes are
 (used as header when tee'ing stdout/stderr outputs)

 - <b>`entrypoint`</b>:  either a `Callable` (function) or `cmd` (binary)

 - <b>`args`</b>:  arguments to each replica

 - <b>`envs`</b>:  env vars to each replica

 - <b>`log_dir`</b>:  directory used to write log files

 - <b>`nprocs`</b>:  number of copies to create (one on each process)

 - <b>`start_method`</b>:  multiprocessing start method (spawn, fork, forkserver)
 ignored for binaries

 - <b>`redirects`</b>:  which std streams to redirect to a log file

 - <b>`tees`</b>:  which std streams to redirect + print to console





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TailLog`
Tails the given log files. The log files do not have to exist when the
`start()` method is called. The tail-er will gracefully wait until the
log files are created by the producer and will tail the contents of the
log files until the `stop()` method is called.


.. warning:: [`TailLog`](./llutil.launcher.elastic_multiprocessing.md#class-taillog) will wait indefinitely for the log file to be created!


Each log file's line will be suffixed with a header of the form: ``[{name}{idx}]:``,
where the `name` is user-provided and `idx` is the index of the log file
in the `log_files` mapping.


Usage:


:
```


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


```
.. note:: Due to buffering log lines between files may not necessarily
 be printed out in order. You should configure your application's
 logger to suffix each log line with a proper timestamp.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    name: str,
    log_files: Dict[int, str],
    dst: <class 'TextIO'>,
    interval_sec: float = 0.1,
    progbar_events=None
)
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `start`

```python
start() → TailLog
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `stop`

```python
stop() → None
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `stopped`

```python
stopped() → bool
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PContext`
The base class that standardizes operations over a set of processes
that are launched via different mechanisms. The name [`PContext`](./llutil.launcher.elastic_multiprocessing.md#class-pcontext)
is intentional to disambiguate with `torch.multiprocessing.ProcessContext`.


.. warning:: stdouts and stderrs should ALWAYS be a superset of
 tee_stdouts and tee_stderrs (respectively) this is b/c
 tee is implemented as a redirect + tail -f <stdout/stderr.log>




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    name: str,
    entrypoint: Union[Callable, str],
    args: Dict[int, Tuple],
    envs: Dict[int, Dict[str, str]],
    stdouts: Dict[int, str],
    stderrs: Dict[int, str],
    tee_stdouts: Dict[int, str],
    tee_stderrs: Dict[int, str],
    error_files: Dict[int, str],
    progbar_events: Dict[str, Event]
)
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `close`

```python
close(death_sig: Optional[Signals] = None, timeout: int = 30) → None
```

Terminates all processes managed by this context and cleans up any
meta resources (e.g. redirect, error_file files).




**Args:**


 - <b>`death_sig`</b>:  Death signal to terminate porcesses.

 - <b>`timeout`</b>:  Time to wait for processes to finish, if process is
 still alive after this time, it will be terminated via SIGKILL.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `pids`

```python
pids() → Dict[int, int]
```

Returns pids of processes mapped by their respective local_ranks




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `start`

```python
start() → None
```

Start processes using parameters defined in the constructor.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `wait`

```python
wait(timeout: float = -1, period: float = 1) → Optional[RunProcsResult]
```

Waits for the specified `timeout` seconds, polling every `period` seconds
for the processes to be done. Returns `None` if the processes are still running
on timeout expiry. Negative timeout values are interpreted as "wait-forever".
A timeout value of zero simply queries the status of the processes (e.g. equivalent
to a poll).


..note: Multiprocesing library registers SIGTERM and SIGINT signal handlers that raise
 `SignalException` when the signals received. It is up to the consumer of the code
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





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiprocessContext`
[`PContext`](./llutil.launcher.elastic_multiprocessing.md#class-pcontext) holding worker processes invoked as a function.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
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
    progbar_events
)
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `close`

```python
close(death_sig: Optional[Signals] = None, timeout: int = 30) → None
```

Terminates all processes managed by this context and cleans up any
meta resources (e.g. redirect, error_file files).




**Args:**


 - <b>`death_sig`</b>:  Death signal to terminate porcesses.

 - <b>`timeout`</b>:  Time to wait for processes to finish, if process is
 still alive after this time, it will be terminated via SIGKILL.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L474"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `pids`

```python
pids() → Dict[int, int]
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `start`

```python
start() → None
```

Start processes using parameters defined in the constructor.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/elastic_multiprocessing.py#L256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `wait`

```python
wait(timeout: float = -1, period: float = 1) → Optional[RunProcsResult]
```

Waits for the specified `timeout` seconds, polling every `period` seconds
for the processes to be done. Returns `None` if the processes are still running
on timeout expiry. Negative timeout values are interpreted as "wait-forever".
A timeout value of zero simply queries the status of the processes (e.g. equivalent
to a poll).


..note: Multiprocesing library registers SIGTERM and SIGINT signal handlers that raise
 `SignalException` when the signals received. It is up to the consumer of the code
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





