<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.hypergraph`








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Task`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(train: bool, tags='*', **kwds)
```











---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Repeat`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(tasks: List[_Task], times: int) → None
```











---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HyperGraph`
HyperGraph is the container for all nodes.
 





<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_node`

```python
add_node(name, node, tags='*')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, device='auto')
```

Ellipsis




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/hypergraph.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(self, tasks, launcher: ElasticLauncher='auto')
```

Ellipsis





