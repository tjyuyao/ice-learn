<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.multiprocessing.spawn`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `start_processes`

```python
start_processes(
    fn,
    args=(),
    nprocs=1,
    join=True,
    daemon=False,
    start_method='spawn'
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `spawn`

```python
spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
```

Spawns ``nprocs`` processes that run ``fn`` with ``args``. 

If one of the processes exits with a non-zero exit status, the remaining processes are killed and an exception is raised with the cause of termination. In the case an exception was caught in the child process, it is forwarded and its traceback is included in the exception raised in the parent process. 



**Args:**
 
 - <b>`fn`</b> (function):  Function is called as the entrypoint of the  spawned process. This function must be defined at the top  level of a module so it can be pickled and spawned. This  is a requirement imposed by multiprocess. 

 The function is called as ``fn(i, *args)``, where ``i`` is  the process index and ``args`` is the passed through tuple  of arguments. 


 - <b>`args`</b> (tuple):  Arguments passed to ``fn``. 
 - <b>`nprocs`</b> (int):  Number of processes to spawn. 
 - <b>`join`</b> (bool):  Perform a blocking join on all processes. 
 - <b>`daemon`</b> (bool):  The spawned processes' daemon flag. If set to True,  daemonic processes will be created. 
 - <b>`start_method`</b> (string):  (deprecated) this method will always use ``spawn``  as the start method. To use a different start method  use ``start_processes()``. 



**Returns:**
 None if ``join`` is ``True``, 
 - <b>`:class`</b>: `~ProcessContext` if ``join`` is ``False`` 




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProcessException`










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProcessRaisedException`
Exception is thrown when the process failed due to exception raised by the code. 







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProcessExitedException`
Exception is thrown when the process failed due to signal or exited with a specific code. 







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProcessContext`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `join`

```python
join(timeout=None)
```

Tries to join one or more processes in this spawn context. If one of them exited with a non-zero exit status, this function kills the remaining processes and raises an exception with the cause of the first process exiting. 

Returns ``True`` if all processes have been joined successfully, ``False`` if there are more processes that need to be joined. 



**Args:**
 
 - <b>`timeout`</b> (float):  Wait this long before giving up on waiting. 



---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `pids`

```python
pids()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SpawnContext`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `join`

```python
join(timeout=None)
```

Tries to join one or more processes in this spawn context. If one of them exited with a non-zero exit status, this function kills the remaining processes and raises an exception with the cause of the first process exiting. 

Returns ``True`` if all processes have been joined successfully, ``False`` if there are more processes that need to be joined. 



**Args:**
 
 - <b>`timeout`</b> (float):  Wait this long before giving up on waiting. 



---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/spawn.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `pids`

```python
pids()
```








