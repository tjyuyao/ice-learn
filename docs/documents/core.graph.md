<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.graph`
contains [`Node`](./core.graph.md#class-node) and [`ExecutableGraph`](./core.graph.md#class-executablegraph).




**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InvalidURIError`
An Exception raised when valid node URI is expected.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StopTask`
An Exception raised to exit current task.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StopAllTasks`
An Exception raised to exit current running.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Node`
This class defines the executable node.


A executable graph is defined by a collection of executable nodes and their dependency relationships. 


A node is executable if it has at least following phases of execution: `forward`, `backward`, `update`. Different subclass of nodes may implement them differently.


This class is designed to be executed easily in batch mode (see [`ExecutableGraph.apply()`](./core.graph.md#method-apply) for details), so that a bunch of nodes can execute together, respecting several synchronization points between phases.


The dependency relationship is determined at runtime by how user access the `graph` argument of `Node.forward()` function. The `graph` argument is actually a cache (a [`GraphOutputCache`](./core.graph.md#class-graphoutputcache) instance) of the graph nodes outputs. The results of precedent nodes will be saved in the cache, so dependents can retrieve them easily.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```








---

#### <kbd>property</kbd> board







---

#### <kbd>property</kbd> device

the assigned device by current launcher.




---

#### <kbd>property</kbd> epoch_size







---

#### <kbd>property</kbd> epoch_steps







---

#### <kbd>property</kbd> global_auto_steps







---

#### <kbd>property</kbd> global_train_epochs







---

#### <kbd>property</kbd> global_train_steps







---

#### <kbd>property</kbd> grad_acc_steps







---

#### <kbd>property</kbd> grad_scaler







---

#### <kbd>property</kbd> launcher







---

#### <kbd>property</kbd> name

the node name in the current activated [`ExecutableGraph`](./core.graph.md#class-executablegraph).




---

#### <kbd>property</kbd> out_dir







---

#### <kbd>property</kbd> run_id







---

#### <kbd>property</kbd> step_mode

whether current task is running by step (True) or by epoch (False).




---

#### <kbd>property</kbd> task







---

#### <kbd>property</kbd> training

whether current task is training.






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward()
```

calculates gradients.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up`

```python
clean_up()
```

an event hook for clean up all resources at switching executable graphs.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dry_run`

```python
dry_run()
```

only update states about progress.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_end`

```python
epoch_end()
```

an event hook for epoch end. (only for epoch mode)




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_start`

```python
epoch_start()
```

an event hook for epoch start. (only for epoch mode)




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward()
```

retrieves forward output in cache or calculates it using `forward_impl` and save the output to the cache. Subclasses should not override this method.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_impl`

```python
forward_impl(cache: "'GraphOutputCache'")
```

forward pass of the node, inputs of current executable graph can be directly retrieved from `graph` argument.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict: 'Dict', strict: 'bool')
```

resumes node state from state_dict.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `move`

```python
move(data, device=None)
```

Moves data to the CPU or the GPU.


If :attr:`device` is `None` and the node has a `device` attribute, then that is the device where the data
is moved. Otherwise, the data is moved according to the `device.type`. If :attr:`data` is a tuple or list,
the function is applied recursively to each of the elements.




**Args:**


 - <b>`data`</b> (torch.Tensor or torch.nn.Module or list or dict):  the data to move.

 - <b>`device`</b> (str or torch.device):  the string or instance of torch.device in which to move the data.




**Returns:**


 - <b>`torch.Tensor or torch.nn.Module`</b>:  the data in the requested device.




**Raises:**


 - <b>`RuntimeError`</b>:  data is not one of torch.Tensor, torch.nn.module, list or dict.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare()
```

an event hook for prepare all resources at switching executable graphs.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict() → Dict
```

returns serialization of current node.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update()
```

update parameters or buffers, e.g. using SGD based optimizer to update parameters. 





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GraphOutputCache`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(egraph: "'ExecutableGraph'") → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__getitem__`

```python
__getitem__(name)
```

Execute node with name `name` if not executed, return the last executed cache else.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clear`

```python
clear()
```

Clear the cache, next calls to `__getitem__` will recalculate.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExecutableGraph`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(hypergraph) → None
```








---

#### <kbd>property</kbd> grad_scaler









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_node`

```python
add_node(node_name, node, tags)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(
    method: 'str',
    *args,
    filter: Callable[[Node], bool] = lambda _: True,,
    **kwds
)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L273"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up_nodes`

```python
clean_up_nodes()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `items`

```python
items()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `iterate`

```python
iterate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare_nodes`

```python
prepare_nodes()
```








