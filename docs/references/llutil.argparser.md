<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.argparser`
This module provides helper functions for commonly used argument processing for functions, 
and a FlexibleArgParser for command line argument parsing. The default singleton of this
argument parser is accessable via ``ice.args``.




**Global Variables**
---------------
- **REQUIRED**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `isa`

```python
isa(obj, types)
```

Helper function: alias for python built-in ``isinstance``.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `as_list`

```python
as_list(maybe_element)
```

Helper function: regularize input into list of element.


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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `as_dict`

```python
as_dict(maybe_element, key)
```

Helper function: regularize input into a dict.


if ``maybe_element`` is not a dict, will return a dict with single
key as ``{key:maybe_element}``, else will return ``maybe_element``.




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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ArgumentMissingError`
Raised when a required argument is missing from command line.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ArgumentTypeError`
Raised when converting an argument failed.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FlexibleArgParser`
A flexible and lightweight argument parser that saves loads of code.


This module works differently compared to python built-in ``argparse`` module.

- It accepts two types of command line arguments, i.e. positional and keyword based (options).

- The keyword based arguments (options) should be specified as ``key=value`` or ``key="value"``.

- The positional arguments is indexed directly using an integer, but this feature is not recommended.




**Example:**



```python
import ice

# same as `python <script>.py 2 k1=4` in shell.
ice.args.parse_args(["2", "k1=4"])

ice.args.setdefault("k1", 8, int)  # This line is optional for this example; setdefault() generally is optional.
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







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parse_args`

```python
parse_args(argv)
```

Manually parse args.




**Args:**


 - <b>`argv`</b> (List[str]):  simillar to `sys.argv[1:]`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `setdefault`

```python
setdefault(key, default, type=None, help='')
```

Set argument value under `key` as `value`, only if original entry does not exists.




**Args:**


 - <b>`key`</b> (int|str):  the keyword.

 - <b>`value`</b>:  default_value to be set when orginal entry does not exists.




**Returns:**

original or updated value.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/argparser.py#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(_FlexibleArgParser__dict={}, **kwds)
```

simillar to dict.update().





