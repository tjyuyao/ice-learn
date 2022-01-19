<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.config`





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_configurable`

```python
is_configurable(cls)
```






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `configurable`

```python
configurable(cls)
```

This wrapper converts ``cls`` to a ``Config`` class which delays the initialization of the original one.  




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_configurable`

```python
make_configurable(*classes)
```






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Config`
Any class converted to ``Config`` instance by ``configurable`` or ``make_configurable`` can store and modify its positional and keyword arguments. The real instance of original class will be created only when config is ``freeze()``-d. The configuration should be ``clone()``-d for new instances. 

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(cls, obj, *args, **kwds) → None
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clone`

```python
clone(deepcopy=True)
```





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `freeze`

```python
freeze()
```





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(explicit={}, **implicit)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
