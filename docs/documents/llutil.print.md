<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/print.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.print`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/print.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `format_size`

```python
format_size(size)
```

Format a byte count as a human readable file size.


```python
format_size(0)
# '0 bytes'
format_size(1)
# '1 byte'
format_size(5)
# '5 bytes'
format_size(1024)
# '1 KiB'
```




