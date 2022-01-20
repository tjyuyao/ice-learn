<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.config`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `configurable`

```python
configurable(cls)
```

This decorator delays the initialization of ``cls`` until ``freeze()``. 



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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_configurable`

```python
make_configurable(*classes)
```

This function converts multiple existing classes to configurables. 



**Note:**

> This have exactly the same effects of decorate each class with `@configurable` when defining the class. Each class only need to be decorated once, extra calling of conversion is ignored and has no side effects. 
>

**Example:**
 

```python
import ice
import torch.nn as nn
ice.make_configurable(nn.Conv2d, nn.Linear)
assert ice.is_configurable(nn.Conv2d)
```
 


---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/config.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `freeze`

```python
freeze(obj, deepcopy=True)
```

freeze configurables recursively. 

**Freezing** is the process of building the configuration into real objects. Original `__init__()` functions of configurable classes declared by ``configurable`` or ``make_configurable`` now will be called recursively to initialize the real instance, also known as the frozen version of a configurable. 



**Args:**
 
 - <b>`obj`</b> (configurable or list/dict of configurables):  the configurable object to be freeze. 
 - <b>`deepcopy`</b> (bool, optional):  copy resources by value. Defaults to True. 



**Returns:**
 Frozen version of the original configurable. 



**Note:**

> Freezing happens in-place, ignoring the returned value is safe. If a user wants to reuse the configuration feature, he can clone() the object before or after frozen with the same effect. 
>

**Example:**
 See examples for ``configurable`` and ``clone``.  




