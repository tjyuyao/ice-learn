<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.hypergraph`






**Global Variables**
---------------
- **TYPE_CHECKING**
- **global_shared_events**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `LoadCheckpointTask`

```python
LoadCheckpointTask(resume_from, strict=False, tags='*')
```

Load checkpoint from a file.




**Args:**


 - <b>`resume_from`</b> (str):  Path to the checkpoint file.

 - <b>`strict`</b> (bool):  If True, raise an exception if the checkpoint file does not exist.

 - <b>`tags`</b> (str):  Tags to load.




**Returns:**


 - <b>`Task`</b>:  Task to load the checkpoint.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `SaveCheckpointTask`

```python
SaveCheckpointTask(save_to=None, tags='*')
```

Save checkpoint to a file.




**Args:**


 - <b>`save_to`</b> (str):  Path to the checkpoint file.

 - <b>`tags`</b> (str):  Tags to save.




**Returns:**


 - <b>`Task`</b>:  Task to save the checkpoint.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ResumeTaskFailed`
raised when task structure does not match during resuming.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Task`
A task is a unit of computation.


It can be a single node, or a graph. A task can be executed by a worker.




**Args:**


 - <b>`node`</b>:  a node or a graph.

 - <b>`name`</b>:  the name of the task.

 - <b>`total_steps`</b>:  the total number of steps to run.

 - <b>`total_epochs`</b>:  the total number of epochs to run.

 - <b>`config`</b>:  a dict of configs.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```








---

#### <kbd>property</kbd> global_auto_epochs







---

#### <kbd>property</kbd> global_auto_steps









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict, dry_run=False)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Repeat`
Repeat a task for a fixed number of times.




**Attributes:**


 - <b>`task`</b> (Task):  Task to repeat.

 - <b>`repeat`</b> (int):  Number of times to repeat the task.

 - <b>`epoch_size`</b> (int):  Number of steps per epoch.

 - <b>`total_steps`</b> (int):  Total number of steps.

 - <b>`total_epochs`</b> (int):  Total number of epochs.

 - <b>`launcher`</b> (Launcher):  Launcher object.

 - <b>`hypergraph`</b> (Hypergraph):  Hypergraph object.

 - <b>`events`</b> (Events):  Events object.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L294"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict, dry_run=False)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L330"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Counter`
Counter object.




**Attributes:**


 - <b>`epochs`</b> (int):  Number of epochs.

 - <b>`steps`</b> (int):  Number of steps.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L337"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```








---

#### <kbd>property</kbd> total









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L341"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__getitem__`

```python
__getitem__(key)
```

Get the value of the counter.




**Args:**


 - <b>`key`</b> (str):  Name of the counter.




**Returns:**


 - <b>`int`</b>:  Value of the counter.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L357"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__setitem__`

```python
__setitem__(key, value)
```

Set the value of the counter.




**Args:**


 - <b>`key`</b> (str):  Name of the counter.

 - <b>`value`</b> (int):  Value of the counter.




**Raises:**


 - <b>`KeyError`</b>:  If the key is not valid.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L378"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GlobalCounters`
Global counters object.




**Attributes:**


 - <b>`epochs`</b> (int):  Number of epochs.

 - <b>`steps`</b> (int):  Number of steps.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    steps: 'Counter' = <core.hypergraph.Counter object at 0x7ff1ec689c40>,
    epochs: 'Counter' = <core.hypergraph.Counter object at 0x7ff1ec6891c0>
) → None
```











---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HyperGraph`
HyperGraph is the container for all nodes.




**Attributes:**


 - <b>`nodes`</b> (dict):  Nodes.

 - <b>`edges`</b> (dict):  Edges.

 - <b>`tasks`</b> (dict):  Tasks.

 - <b>`launchers`</b> (dict):  Launchers.

 - <b>`global_counters`</b> (GlobalCounters):  Global counters.

 - <b>`resume_from`</b> (str):  Path to the checkpoint file.

 - <b>`resume_tags`</b> (str):  Tags to load.

 - <b>`save_to`</b> (str):  Path to the checkpoint file.

 - <b>`save_tags`</b> (str):  Tags to save.

 - <b>`strict`</b> (bool):  If True, raise an exception if the checkpoint file does not exist.

 - <b>`dry_run`</b> (bool):  If True, do not save the checkpoint.

 - <b>`verbose`</b> (bool):  If True, print the progress.

 - <b>`logger`</b> (Logger):  Logger.




**Raises:**


 - <b>`ValueError`</b>:  If the tags are not valid.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    autocast_enabled=False,
    autocast_dtype=None,
    grad_scaler: 'Union[bool, GradScaler]' = None
) → None
```








---

#### <kbd>property</kbd> launcher

Get the launcher.




**Returns:**


 - <b>`ElasticLauncher`</b>:  Launcher.




**Raises:**


 - <b>`ValueError`</b>:  If the launcher is not valid.






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__getitem__`

```python
__getitem__(uid) → Node
```

Get a node by uid.




**Args:**


 - <b>`uid`</b> (str):  Uid.




**Returns:**


 - <b>`Node`</b>:  Node.




**Raises:**


 - <b>`ValueError`</b>:  If the uid is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L555"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(name, node: 'Node', tags='*')
```

Add a node.




**Args:**


 - <b>`name`</b> (str):  Name.

 - <b>`node`</b> (Node):  Node.

 - <b>`tags`</b> (str):  Tags.




**Returns:**


 - <b>`Node`</b>:  Node.




**Raises:**


 - <b>`ValueError`</b>:  If the name is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L472"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backup_source_files`

```python
backup_source_files(entrypoint: 'str')
```

Backup source files.




**Args:**


 - <b>`entrypoint`</b> (str):  Entrypoint.




**Raises:**


 - <b>`ValueError`</b>:  If the entrypoint is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L936"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `exec_tasks`

```python
exec_tasks(tasks, launcher: 'ElasticLauncher')
```

Execute the tasks.




**Args:**


 - <b>`tasks`</b> (List[Task]):  Tasks to execute.

 - <b>`launcher`</b> (ElasticLauncher):  Launcher.




**Returns:**


 - <b>`List[Task]`</b>:  Tasks executed.




**Raises:**


 - <b>`ValueError`</b>:  If the tasks are not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L447"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_autocast`

```python
init_autocast(
    autocast_enabled=True,
    autocast_dtype=None,
    grad_scaler: 'Union[bool, GradScaler]' = None
)
```

Initialize autocast.




**Args:**


 - <b>`autocast_enabled`</b> (bool):  If True, enable autocast.

 - <b>`autocast_dtype`</b> (str):  Data type to cast the gradients to.

 - <b>`grad_scaler`</b> (GradScaler):  Gradient scaler.




**Raises:**


 - <b>`ValueError`</b>:  If the autocast_dtype is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L494"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_grad_scaler`

```python
init_grad_scaler(self, grad_scaler: Union[bool, GradScaler]=False, *, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L464"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_autocast_enabled`

```python
is_autocast_enabled() → bool
```

Check if autocast is enabled.




**Returns:**


 - <b>`bool`</b>:  If True, autocast is enabled.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_grad_scaler_enabled`

```python
is_grad_scaler_enabled() → bool
```

Check if the gradient scaler is enabled.




**Returns:**


 - <b>`bool`</b>:  If True, the gradient scaler is enabled.




**Raises:**


 - <b>`ValueError`</b>:  If the grad_scaler is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L896"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_checkpoint`

```python
load_checkpoint(resume_from, strict=False, tags='*')
```

Load the checkpoint.




**Args:**


 - <b>`resume_from`</b> (str):  Path to the checkpoint.

 - <b>`strict`</b> (bool):  Whether to check the keys.

 - <b>`tags`</b> (str):  Tags to load.




**Raises:**


 - <b>`ValueError`</b>:  If the resume_from is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L688"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `print_forward_output`

```python
print_forward_output(
    *nodenames,
    every=1,
    total=None,
    tags: 'List[str]' = '*',
    train_only=True,
    localrank0_only=True
)
```

Print forward output.




**Args:**


 - <b>`nodenames`</b> (str):  Node names.

 - <b>`every`</b> (int):  Print every.

 - <b>`total`</b> (int):  Total.

 - <b>`tags`</b> (List[str]):  Tags.

 - <b>`train_only`</b> (bool):  Train only.

 - <b>`localrank0_only`</b> (bool):  Local rank 0 only.




**Raises:**


 - <b>`ValueError`</b>:  If the nodenames is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L579"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove`

```python
remove(query)
```

Remove a node.




**Args:**


 - <b>`query`</b> (str):  Query.




**Raises:**


 - <b>`ValueError`</b>:  If the query is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L825"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, devices='auto', run_id: str='none', out_dir: str=None, resume_from: str=None, seed=0)
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L825"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, launcher: ElasticLauncher=None, run_id: str='none', out_dir: str=None, resume_from: str=None, seed=0)
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L825"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, devices='auto', run_id='none', nnodes='1:1', dist_backend='auto', monitor_interval=5, node_rank=0, master_addr='127.0.0.1', master_port=None, redirects='2', tee='1', out_dir=None, resume_from=None, seed=0, role='default', max_restarts=0, omp_num_threads=1, start_method='spawn')
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L825"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, devices='auto', run_id='none', nnodes='1:1', dist_backend='auto', monitor_interval=5, rdzv_endpoint='', rdzv_backend='static', rdzv_configs='', standalone=False, redirects='2', tee='1', out_dir=None, resume_from=None, seed=0, role='default', max_restarts=0, omp_num_threads=1, start_method='spawn')
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L862"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_checkpoint`

```python
save_checkpoint(save_to=None, tags='*')
```

Save the checkpoint.




**Args:**


 - <b>`save_to`</b> (str):  Path to save the checkpoint.

 - <b>`tags`</b> (str):  Tags to save.




**Returns:**


 - <b>`str`</b>:  Path to the checkpoint.




**Raises:**


 - <b>`ValueError`</b>:  If the save_to is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L590"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `select_egraph`

```python
select_egraph(query) → ExecutableGraph
```

Select an executable graph.




**Args:**


 - <b>`query`</b> (str):  Query.




**Returns:**


 - <b>`ExecutableGraph`</b>:  Executable graph.




**Raises:**


 - <b>`ValueError`</b>:  If the query is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L640"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `select_nodes`

```python
select_nodes(*query)
```

Select nodes.




**Args:**


 - <b>`query`</b> (str):  Query.




**Returns:**


 - <b>`list`</b>:  Nodes.




**Raises:**


 - <b>`ValueError`</b>:  If the query is not valid.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L532"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_gradient_accumulate`

```python
set_gradient_accumulate(every=1)
```

Set the gradient accumulate steps.




**Args:**


 - <b>`every`</b> (int):  Gradient accumulate steps.




**Raises:**


 - <b>`ValueError`</b>:  If the every is not valid.





