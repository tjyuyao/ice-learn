<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.hypergraph`






**Global Variables**
---------------
- **TYPE_CHECKING**
- **global_shared_events**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `LoadCheckpointTask`

```python
LoadCheckpointTask(resume_from, strict=False, tags='*')
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `SaveCheckpointTask`

```python
SaveCheckpointTask(save_to=None, tags='*')
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ResumeTaskFailed`
raised when task structure does not match during resuming.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Task`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```








---

#### <kbd>property</kbd> global_auto_epochs







---

#### <kbd>property</kbd> global_auto_steps









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict, dry_run=False)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Repeat`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict, dry_run=False)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Counter`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```








---

#### <kbd>property</kbd> total










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GlobalCounters`
GlobalCounters(steps: 'Counter' = <core.hypergraph.Counter object at 0x7f95070425e0>, epochs: 'Counter' = <core.hypergraph.Counter object at 0x7f9507042070>)




<a href="https://github.com/tjyuyao/ice-learn/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    steps: 'Counter' = <core.hypergraph.Counter object at 0x7f95070425e0>,
    epochs: 'Counter' = <core.hypergraph.Counter object at 0x7f9507042070>
) → None
```











---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HyperGraph`
HyperGraph is the container for all nodes.
 





<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L368"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(name, node: 'Node', tags='*')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backup_source_files`

```python
backup_source_files(entrypoint: 'str')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L635"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `exec_tasks`

```python
exec_tasks(tasks, launcher: 'ElasticLauncher')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_autocast`

```python
init_autocast(
    autocast_enabled=True,
    autocast_dtype=None,
    grad_scaler: 'Union[bool, GradScaler]' = None
)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_grad_scaler`

```python
init_grad_scaler(self, grad_scaler: Union[bool, GradScaler]=False, *, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_autocast_enabled`

```python
is_autocast_enabled() → bool
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L358"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_grad_scaler_enabled`

```python
is_grad_scaler_enabled() → bool
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L605"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_checkpoint`

```python
load_checkpoint(resume_from, strict=False, tags='*')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove`

```python
remove(query)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, devices='auto', run_id: str='none', out_dir: str=None, resume_from: str=None, seed=0)
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, launcher: ElasticLauncher=None, run_id: str='none', out_dir: str=None, resume_from: str=None, seed=0)
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, devices='auto', run_id='none', nnodes='1:1', dist_backend='auto', monitor_interval=5, node_rank=0, master_addr='127.0.0.1', master_port=None, redirects='2', tee='3', out_dir=None, resume_from=None, seed=0, role='default', max_restarts=0, omp_num_threads=1, start_method='spawn')
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, devices='auto', run_id='none', nnodes='1:1', dist_backend='auto', monitor_interval=5, rdzv_endpoint='', rdzv_backend='static', rdzv_configs='', standalone=False, redirects='2', tee='3', out_dir=None, resume_from=None, seed=0, role='default', max_restarts=0, omp_num_threads=1, start_method='spawn')
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L581"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_checkpoint`

```python
save_checkpoint(save_to=None, tags='*')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `select_egraph`

```python
select_egraph(query) → ExecutableGraph
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `select_nodes`

```python
select_nodes(*query)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_gradient_accumulate`

```python
set_gradient_accumulate(every=1)
```








