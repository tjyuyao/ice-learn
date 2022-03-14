<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/collections.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.collections`








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/collections.py#L1"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Dict`
access dict values as attributes.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/collections.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Counter`
count values by group.


**Features:**

- Get or set values using dictionary or attribute interface.

- Returns a zero count for missing items instead of raising a KeyError.

- a `total()` function that sums all values.




**Example:**



```python
import ice
cnt = ice.Counter()
assert 0 == cnt['x']
assert 0 == cnt.x
cnt.x += 1
assert 1 == cnt['x']
assert 1 == cnt.x
cnt['y'] += 1
assert 2 == cnt.total()
```






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/collections.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `total`

```python
total()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/collections.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConfigDict`
stores multi-level configurations easily.


**Features:**

- Get or set values using dictionary or attribute interface.

- Create empty dict for intermediate items instead of raising a KeyError.




**Example:**



```python
import ice
_C = ice.ConfigDict()
_C.PROPERTY1 = 1
_C.GROUP1.PROPERTY1 = 2
```







