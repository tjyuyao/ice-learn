<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/logger.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.logger`
logging utilities.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/logger.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_logger`

```python
get_logger(name: Optional[str] = None)
```

set up a simple logger that writes into stderr. 


The loglevel is fetched from the LOGLEVEL
env. variable or WARNING as default. The function will use the
module name of the caller if no name is provided.




**Args:**


 - <b>`name`</b>:  Name of the logger. If no name provided, the name will
 be derived from the call stack.





