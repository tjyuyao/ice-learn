<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.config`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `configurable`

```python
configurable(cls)
```

This decorator delays the initialization of `cls` until [`freeze()`](./llutil.config.md#function-freeze).




**Returns:**

 decorated class which is now configurable.




**Example:**



```python
import ice

@ice.configurable
class AClass:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

# partial initialization.
i = AClass(b=0)

# alter positional and keyword arguments afterwards.
i[0] = 2
i['b'] = 1
i.update({'c': 3, 'd': 4})
i.update(d=5)

# unfrozen configurable can be printed as a legal construction python statement.
assert repr(i) == "AClass(a=2, b=1, c=3, d=5)"

# real initialization of original object.
i.freeze()
assert i.a == 2 and i.b == 1
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_configurable`

```python
is_configurable(cls) → bool
```

check if a class or an object is configurable.




**Returns:**

 bool




**Example:**



```python
import ice
import torch.nn as nn
ice.make_configurable(nn.Conv2d, nn.Linear)
assert ice.is_configurable(nn.Conv2d)
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `has_builder`

```python
has_builder(obj) → bool
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `frozen`

```python
frozen(obj)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_configurable`

```python
make_configurable(*classes)
```

This function converts multiple existing classes to configurables.




**Note:**

> This have exactly the same effects of decorate each class with `@configurable` when defining the class.
 Each class only need to be decorated once, extra calling of conversion is ignored and has no side effects.




**Example:**



```python
import ice
import torch.nn as nn
ice.make_configurable(nn.Conv2d, nn.Linear)
assert ice.is_configurable(nn.Conv2d)
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clone`

```python
clone(obj, deepcopy=True)
```

clone configurables, containers, and ordinary objects recursively.




**Args:**


 - <b>`obj`</b> (configurable or list/dict of configurables):  the configurable object to be cloned.

 - <b>`deepcopy`</b> (bool, optional):  copy resources by value. Defaults to True.




**Returns:**

Unfrozen copy of the original configurable.


```python
import ice
import torch.nn as nn
ice.make_configurable(nn.Conv2d, nn.Linear)

convcfg = nn.Conv2d(16, 8)

conv1x1 = convcfg.clone()  # or ice.clone(convcfg)
conv1x1['kernel_size'] = 1
conv1x1.freeze()  # or ice.freeze(conv1x1)
assert conv1x1.kernel_size == (1, 1)

conv3x3 = convcfg.clone()
conv3x3['kernel_size'] = 3
conv3x3.freeze()
assert conv3x3.kernel_size == (3, 3)
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `freeze`

```python
freeze(obj)
```

freeze configurables recursively.


**Freezing** is the process of building the configuration into real objects.
Original `__init__()` functions of configurable classes declared by [`configurable`](./llutil.config.md#function-configurable)
or [`make_configurable`](./llutil.config.md#function-make_configurable) now will be called recursively to initialize the real instance,
also known as the frozen version of a configurable.




**Args:**


 - <b>`obj`</b> (configurable or list/dict of configurables):  the configurable object to be freeze.




**Returns:**

Frozen version of the original configurable.




**Note:**

>Freezing happens in-place, ignoring the returned value is safe.
If a user wants to reuse the configuration feature, he can clone() the
object before or after frozen with the same effect.




**Example:**

See examples for [`configurable`](./llutil.config.md#function-configurable) and [`clone`](./llutil.config.md#function-clone).





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `objattr`

```python
objattr(obj, attrname)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Configurable`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L452"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `auto_freeze`

```python
auto_freeze()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clone`

```python
clone(deepcopy=True)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L425"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `extra_repr`

```python
extra_repr()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `freeze`

```python
freeze()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_params`

```python
update_params(*args, **kwds)
```








