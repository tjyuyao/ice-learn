<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/board.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.board`








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/board.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoardWriter`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/board.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    log_dir=None,
    comment='',
    purge_step=None,
    max_queue=10,
    flush_secs=120,
    filename_suffix=''
)
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/board.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_hparams`

```python
add_hparams(hparam_domain_discrete=None)
```

Add a set of hyperparameters to be compared in TensorBoard.




**Args:**


 - <b>`hparam_domain_discrete`</b>:  (Optional[Dict[str, List[Any]]]) A dictionary that
 contains names of the hyperparameters and all discrete values they can hold




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/board.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_scalar`

```python
add_scalar(
    tag,
    scalar_value,
    global_step=None,
    walltime=None,
    new_style=False,
    double_precision=False
)
```








