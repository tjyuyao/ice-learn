<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.graph`
An executable configuration graph.




**Note:**

> We describe the concept of this core module in following few lines and show some pesudo-codes. This is very close to but not the same as the real code.


An acyclic directed hypergraph $G$ consists of a set of vertices $V$ and a set of hyperarcs $H$, where a hyperarc is a pair $<X, Y>$ , $X$ and $Y$ non empty subset of $V$.


We have a tag system that split the vertices $V$ into maybe overlapping subsets $V_i$, that each of which is a degenerated hypergraph $G_i$ that only consists of vertices $V_i$ and a set of hyperarcs $H_i$ so that each hyperarc is a pair $<x, Y>$, where $x \in V_i$ and $Y \subset V_i$. We call tails $x$ as producers and heads $Y$ as consumers in each hyperarc, this states the dependencies.


User defines a vertice (`Node` in the code) by specify a computation process $f$ (`forward` in the code) and the resources $R$ (`Dataset`s, `nn.Module`s, imperatively programmed function definitions such as losses and metrics, etc.) needed by it.


```python
vertice_1 = Node(
     name = "consumer_node_name",
     resources = ...,
     forward = lambda n, x: do_something_with(n.resources, x["producer_node_name"]),
     tags = ["group1", "group2"],
)
```


A longer version of `forward` parameter that corresponds to the previous notation would be `forward = lambda self, V_i: do_something_with(self.resources, V_i["x"])`,  but we will stick to the shorter version in the code.


So at the time of configuration, we are able to define every material as a node, and the name of nodes can be duplicated, i.e. multiple $x\in V$ can have the same identifier, as long as they does not have the same tag $i$ that selects $V_i$. The tags mechanism is flexible. Every node can have multiple of them, and multiple tags can be specified so that a union of subsets will be retrieved. If no tag is specified for a node, a default tag `*` will be used and a retrival will always include the `*` group.


```python
hyper_graph = HyperGraph([
     vertice_1,
     vertice_2,
     ...,
     vertice_n,
])

activated_graph = hyper_graph["group1", "group3", "group5"]
freeze_and_execute(activated_graph)
```






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InvalidURIError`
An Exception raised when valid node URI is expected.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StopTask`
An Exception raised to exit current task.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Node`







---

#### <kbd>property</kbd> device







---

#### <kbd>property</kbd> name

Returns the node name in the current activated ``ExecutableGraph``.




**Returns:**


 - <b>`str|None`</b>:  the name specified by `ice.add_...(name=...)`.




---

#### <kbd>property</kbd> step_mode







---

#### <kbd>property</kbd> training







---

#### <kbd>property</kbd> uris

Returns the node URIs in the current ``HyperGraph``.




**Returns:**


 - <b>`List[str]`</b>:  ["{tag}/{name}"], each as a unique identifier of this node.






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up`

```python
clean_up()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dry_run`

```python
dry_run()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_end`

```python
epoch_end()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_start`

```python
epoch_start()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `interrupt`

```python
interrupt()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare()
```

Prepare all resources, including moving tensors to GPU.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NodeOutputCache`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__getitem__`

```python
__getitem__(name)
```

Execute node with name ``name`` if not executed, return the last executed cache else.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clear`

```python
clear()
```

Clear the cache, next calls to ``__getitem__`` will recalculate.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExecutableGraph`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_node`

```python
add_node(node_name, node, group_names)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(method: str)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_up`

```python
clean_up()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `iterate`

```python
iterate(hyper_graph)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `partial_apply`

```python
partial_apply(method: str, filter: Callable[[Node], bool] = lambda _: True)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/graph.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare()
```








