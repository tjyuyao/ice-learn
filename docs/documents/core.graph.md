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

the board writer of current task.




---

#### <kbd>property</kbd> device

the assigned device by current launcher.




---

#### <kbd>property</kbd> epoch_size

the size of current epoch.




---

#### <kbd>property</kbd> epoch_steps

the steps of current epoch.




---

#### <kbd>property</kbd> global_auto_steps

the global steps of current task.




---

#### <kbd>property</kbd> global_train_epochs

the global train epochs of current task.




---

#### <kbd>property</kbd> global_train_steps

the global train steps of current task.




---

#### <kbd>property</kbd> grad_acc_steps

the grad accumulator steps of current task.




---

#### <kbd>property</kbd> grad_scaler

the grad scaler of current task.




---

#### <kbd>property</kbd> launcher

the current launcher.




---

#### <kbd>property</kbd> name

the node name in the current activated [`ExecutableGraph`](./core.graph.md#class-executablegraph).




---

#### <kbd>property</kbd> out_dir

the output directory of current task.




---

#### <kbd>property</kbd> run_id

the run id of current task.




---

#### <kbd>property</kbd> step_mode

whether current task is running by step (True) or by epoch (False).




---

#### <kbd>property</kbd> task

the current task.




---

#### <kbd>property</kbd> training

whether current task is training.






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward()
```

calculates gradients.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up`

```python
clean_up()
```

an event hook for clean up all resources at switching executable graphs.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dry_run`

```python
dry_run()
```

only update states about progress.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_end`

```python
epoch_end()
```

an event hook for epoch end. (only for epoch mode)




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_start`

```python
epoch_start()
```

an event hook for epoch start. (only for epoch mode)




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward()
```

retrieves forward output in cache or calculates it using `forward_impl` and save the output to the cache. Subclasses should not override this method.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_impl`

```python
forward_impl(cache: "'GraphOutputCache'")
```

forward pass of the node, inputs of current executable graph can be directly retrieved from `graph` argument.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict: 'Dict', strict: 'bool')
```

resumes node state from state_dict.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare()
```

an event hook for prepare all resources at switching executable graphs.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict() → Dict
```

returns serialization of current node.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update()
```

update parameters or buffers, e.g. using SGD based optimizer to update parameters. 





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GraphOutputCache`
a cache for storing and searching forward outputs of nodes.


This class is used to store and search forward outputs of nodes.




**Attributes:**


 - <b>`cache`</b> (dict):  a dict for storing forward outputs.

 - <b>`egraph`</b> (ExecutableGraph):  the executable graph.

 - <b>`data`</b> (Dict[str, torch.Tensor]):  the cache.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(egraph: "'ExecutableGraph'") → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__getitem__`

```python
__getitem__(name)
```

Execute node with name `name` if not executed, return the last executed cache else.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clear`

```python
clear()
```

Clear the cache, next calls to `__getitem__` will recalculate.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L249"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExecutableGraph`
an executable graph.


This class is used to execute nodes in a graph.




**Attributes:**


 - <b>`hypergraph`</b> (HyperGraph):  the hypergraph.

 - <b>`nodes`</b> (Dict[str, Node]):  a dict for storing nodes.

 - <b>`nodes_tags`</b> (Dict[str, str]):  a dict for storing tags of nodes.

 - <b>`nodes_names`</b> (Dict[str, str]):  a dict for storing names of nodes.

 - <b>`cache`</b> (GraphOutputCache):  a cache for storing and searching forward outputs of nodes.

 - <b>`task`</b>:  the task of the graph.

 - <b>`losses`</b>:  the losses of the graph.

 - <b>`total_loss`</b>:  the total loss of the graph.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(hypergraph) → None
```








---

#### <kbd>property</kbd> grad_scaler









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_node`

```python
add_node(node_name, node, tags)
```

add a node to the graph.




**Args:**


 - <b>`node_name`</b> (str):  the name of the node.

 - <b>`node`</b> (Node):  the node.

 - <b>`tags`</b> (List[str]):  the tags of the node.




**Raises:**


 - <b>`RuntimeError`</b>:  the node is not a node of the hypergraph.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(
    method: 'str',
    *args,
    filter: Callable[[Node], bool] = lambda _: True,,
    **kwds
)
```

apply `method` to all nodes in the graph.




**Args:**


 - <b>`method`</b> (str):  the method name.

 - <b>`*args`</b>:  the arguments of the method.

 - <b>`filter`</b> (Callable[[Node], bool]):  the filter function.

 - <b>`**kwds`</b>:  the keyword arguments of the method.




**Returns:**


 - <b>`List[Any]`</b>:  the return values of the method.




**Raises:**


 - <b>`RuntimeError`</b>:  the method is not found.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up_nodes`

```python
clean_up_nodes()
```

clean up all nodes in the graph.


This method is called after the graph is executed.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `items`

```python
items()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L346"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `iterate`

```python
iterate()
```

iterate all nodes in the graph.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare_nodes`

```python
prepare_nodes()
```

prepare all nodes in the graph.


This method is called before the graph is executed.





