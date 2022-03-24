<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/optim.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.optim`






**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/optim.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Optimizer`
Optimizer configuration API for ice-learn.


This is an extension of `torch.optim.Optimizer` that:

- allows the user to update the optimizer states using `ice.DictProcessor`,

- leverages `torch.ZeroRedundancyOptimizer` inside for memory efficient distributed training,

- is able to accumulate gradients for simulating large batch size,

- etc.


**Inspired by:**

- https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer

- https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html

- https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/optim.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/optim.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/optim.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict(rank)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/optim.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(self, grad_scaler: GradScaler, grad_acc_steps: int, *, current_epoch, epoch_steps, global_steps, epoch_size)
```

Ellipsis





