<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.argparser`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `isa`

```python
isa(obj, types)
```

alias for python built-in ``isinstance``. 




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FlexibleArgParser`
A flexible and lightweight argument parser that saves loads of code. 

This module works differently compared to python built-in ``argparse`` module. 
- It accepts two types of command line arguments, i.e. positional and keyword based (options). 
- The keyword based arguments (options) should be specified as ``key=value`` or ``key="value"``. 
- The positional arguments can be specified same as ``argparse`` would expect. 

**Example 1:** 

```python
import ice

# same as `python <script>.py 2 workers=4` in shell.
ice.args.parse_args(["2", "workers=4"])

# get 0-th positional argument, as int, default to 4.
batchsize = ice.args.get(0, int, 4)  

# get option named "workers", as int, default to 4.
num_workers = ice.args.get("workers", int, 4)

# Following lines have same effect but when default value is invalid will produce error converting `None` into `int`. You can set default value beforehand use ``ice.args.setdefault()`` to avoid this.
batchsize = int(ice.args[0])
num_workers = int(ice.args["workers"])

# Following line also works, but only for keyword arguments, as integer literal is not a legal attribute name.
num_workers = int(ice.args.workers)
```
 **Example 2:** 

```python
ice.args.parse_args(["2", "k1=4"])
assert len(ice.args) == 2
assert 2 == ice.args.get(0, int, 4)
assert 4 == ice.args.get("k1", int, 8)
assert 4 == int(ice.args["k1"])
assert 4 == int(ice.args.k1)
ice.args.setdefault("k2", 8)

assert 8 == int(ice.args.k2)

ice.args.setdefault("k1", 8)
assert 4 == int(ice.args.k1)

del ice.args["k1"]
assert "k1" not in ice.args
ice.args.setdefault("k1", 8)
assert "k1" in ice.args
assert 8 == int(ice.args.k1)

ice.args.update(k2=0)
ice.args.update({0: 0})
assert 0 == ice.args.get(0, int, 4)
assert 0 == ice.args.get("k2", int, 4)
```
 




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(key, type=None, value=None)
```

get argument as ``type`` with default ``value``. 



**Args:**
 
 - <b>`key`</b> (int|str):  ``int`` for positional argument and ``str`` for options. 
 - <b>`type`</b> (Type, optional):  If specified, the type of argument stored will be casted to ``type``. Default command line arguments are ``str``s. 
 - <b>`value`</b> (Any, optional):  If key not found, will return ``value``. Defaults to None. 



**Returns:**
 
 - <b>`type`</b>:  specific argument value. 



---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parse_args`

```python
parse_args(argv)
```

Manually parse args. 



**Args:**
 
 - <b>`argv`</b> (List[str]):  simillar to `sys.argv[1:]`. 



---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `setdefault`

```python
setdefault(key, value)
```

Set argument value under `key` as `value`, only if original entry does not exists. 



**Args:**
 
 - <b>`key`</b> (int|str):  the keyword. 
 - <b>`value`</b>:  default_value to be set when orginal entry does not exists. 



**Returns:**
 original or updated value. 



---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(*args, **kwargs)
```

simillar to dict.update().  






