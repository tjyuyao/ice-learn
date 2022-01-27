<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.graph`
contains [`Node`](./core.graph.md#class-node) and [`ExecutableGraph`](./core.graph.md#class-executablegraph).






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InvalidURIError`
An Exception raised when valid node URI is expected.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StopTask`
An Exception raised to exit current task.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Node`
This class defines the executable node.


A executable graph is defined by a collection of executable nodes and their dependency relationships. 


A node is executable if it has at least following phases of execution: `forward`, `backward`, `update`. Different subclass of nodes may implement them differently.


This class is designed to be executed easily in batch mode (see [`ExecutableGraph.apply()`](./core.graph.md#method-apply) for details), so that a bunch of nodes can execute together, respecting several synchronization points between phases.


The dependency relationship is determined at runtime by how user access the `graph` argument of `Node.forward()` function. The `graph` argument is actually a cache (a [`GraphOutputCache`](./core.graph.md#class-graphoutputcache) instance) of the graph nodes outputs. The results of precedent nodes will be saved in the cache, so dependents can retrieve them easily.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__freeze__`

```python
__freeze__(
    forward: Callable[[ForwardRef('Node'), ForwardRef('GraphOutputCache')], Any] = None,
    **resources
) → None
```

initialize the node.




**Args:**


 - <b>`forward`</b> (Callable[[self, x:[`GraphOutputCache`](./core.graph.md#class-graphoutputcache)], Any], optional):  if specified, will override the original forward method.

 - <b>`**resources`</b>:  resources will be updated into the attributes of Node.





---

#### <kbd>property</kbd> device

the assigned device by current launcher.




---

#### <kbd>property</kbd> name

the node name in the current activated [`ExecutableGraph`](./core.graph.md#class-executablegraph).




---

#### <kbd>property</kbd> step_mode

whether current task is running by step (True) or by epoch (False).




---

#### <kbd>property</kbd> training

whether current task is training.




---

#### <kbd>property</kbd> uris

the node URIs `<tag/name>` in the current [`HyperGraph`](./core.hypergraph.md#class-hypergraph).






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward()
```

calculates gradients.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up`

```python
clean_up()
```

an event hook for clean up all resources at switching executable graphs, e.g. clear device memory, closing files, etc.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dry_run`

```python
dry_run()
```

only update states about progress.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_end`

```python
epoch_end()
```

an event hook for epoch end. (only for epoch mode)




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_start`

```python
epoch_start()
```

an event hook for epoch start. (only for epoch mode)




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(graph: 'GraphOutputCache')
```

calculates forward pass results of the node, inputs of current executable graph can be directly retrieved from `graph` argument.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(state_dict)
```

resumes node state from state_dict.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare()
```

an event hook for prepare all resources at switching executable graphs, e.g. moving models to device, initialize dataloaders, etc.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict()
```

returns serialization of current node.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update()
```

update parameters or buffers, e.g. using SGD based optimizer to update parameters. 





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GraphOutputCache`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(graph: 'ExecutableGraph') → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__getitem__`

```python
__getitem__(name)
```

Execute node with name `name` if not executed, return the last executed cache else.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clear`

```python
clear()
```

Clear the cache, next calls to `__getitem__` will recalculate.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExecutableGraph`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_node`

```python
add_node(node_name, node, group_names)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(
    method: str,
    *args,
    filter: Callable[[Node], bool] = lambda _: True,,
    **kwds
)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up_nodes`

```python
clean_up_nodes()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `iterate`

```python
iterate(hyper_graph)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare_nodes`

```python
prepare_nodes()
```








