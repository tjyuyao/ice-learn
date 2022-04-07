<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.argparser`
parse arguments for functions and command line.


This module provides helper functions for commonly used argument processing for functions, 
and a FlexibleArgParser for command line argument parsing. The default singleton of this
argument parser is accessable via `ice.args`.




**Global Variables**
---------------
- **REQUIRED**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `isa`

```python
isa(obj, types)
```

an alias for python built-in `isinstance`.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_scalar`

```python
parse_scalar(obj)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `as_list`

```python
as_list(maybe_element)
```

helps to regularize input into list of element.


No matter what is input, will output a list for your iteration.


**Basic Examples:**


```python
assert as_list("string") == ["string"]
assert as_list(["string", "string"]) == ["string", "string"]
assert as_list(("string", "string")) == ["string", "string"]
assert as_list([["string", "string"]]) == ["string", "string"]
```

**An Application Example:**


```python
def func(*args):
    return as_list(args)

assert func("a", "b") == ["a", "b"]
assert func(["a", "b"]) == ["a", "b"]
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `as_dict`

```python
as_dict(maybe_element, key)
```

helps to regularize input into a dict.


if `maybe_element` is not a dict, will return a dict with single
key as ``{key:maybe_element}``, else will return `maybe_element`.




**Args:**


 - <b>`maybe_element`</b>:  a dict or any object.

 - <b>`key `</b>:  the sole key.




**Returns:**


 - <b>`dict`</b>:  ensures to be a dict.




**Example:**



```python
assert as_dict({"k": "v"}, "k") == {"k": "v"}
assert as_dict("v", "k") == {"k": "v"}
```




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_seq_of`

```python
is_seq_of(seq, expected_type, seq_type=None)
```

Check whether it is a sequence of some type.




**Args:**


 - <b>`seq`</b> (Sequence):  The sequence to be checked.

 - <b>`expected_type`</b> (type):  Expected type of sequence items.

 - <b>`seq_type`</b> (type, optional):  Expected sequence type.




**Returns:**


 - <b>`bool`</b>:  Whether the sequence is valid.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_list_of`

```python
is_list_of(seq, expected_type)
```

Check whether it is a list of some type.


A partial method of :func:`is_seq_of`.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_tuple_of`

```python
is_tuple_of(seq, expected_type)
```

Check whether it is a tuple of some type.


A partial method of :func:`is_seq_of`.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_hostname`

```python
get_hostname()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ArgumentMissingError`
Raised when a required argument is missing from command line.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ArgumentTypeError`
Raised when converting an argument failed.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FlexibleArgParser`
A flexible and lightweight argument parser that saves loads of code.


This module works differently compared to python built-in `argparse` module.

- It accepts two types of command line arguments, i.e. positional and keyword based (options).

- The keyword based arguments (options) should be specified as ``key=value`` or ``key="value"``.

- The positional arguments is indexed directly using an integer, but this feature is not recommended.




**Example:**



```python
import ice

# same as `python <script>.py 2 k1=4` in shell.
ice.args.parse_args(["2", "k1=4"])

# setdefault() generally is optional.
ice.args.setdefault("k1", 8, int)
ice.args.setdefault("k2", 8)

assert len(ice.args) == 3
assert 2 == int(ice.args[0])  # default type is str.
assert 4 == ice.args["k1"]  # as setdefault specified a type, here a conversion is not needed.
assert 4 == ice.args.k1  # attribute also works.
assert 8 == ice.args.k2  # use default value.

ice.args["k1"] = 1
ice.args.k3 = 1
ice.args.update(k2=0)
ice.args.update({0: -1})
assert -1 == ice.args[0]
assert  1 == ice.args["k3"]
assert  0 == ice.args.k2
```



**Note:**

> If you manually call `parse_args()`, call it before `setdefault()`.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__() → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(key)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `hparam_dict`

```python
hparam_dict() → Dict
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parse_args`

```python
parse_args(argv)
```

Manually parse args.




**Args:**


 - <b>`argv`</b> (List[str]):  simillar to `sys.argv[1:]`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set`

```python
set(key, value, hparam=False)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `setdefault`

```python
setdefault(key, default, _type=<class 'str'>, hparam=False, help='')
```

Set argument value under `key` as `value`, only if original entry does not exists.




**Args:**


 - <b>`key`</b> (int|str):  the keyword.

 - <b>`value`</b>:  default_value to be set when orginal entry does not exists.




**Returns:**

original or updated value.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(_FlexibleArgParser__dict={}, **kwds)
```

simillar to dict.update().





