<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/loss.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.loss`








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/loss.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LossMode`
An enumeration.








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/loss.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LossNode`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/loss.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds)
```








---

#### <kbd>property</kbd> board

the board writer of current task.




---

#### <kbd>property</kbd> device

the assigned device by current launcher.




---

#### <kbd>property</kbd> epoch_size

the size of current epoch.




---

#### <kbd>property</kbd> epoch_steps

the steps of current epoch.




---

#### <kbd>property</kbd> global_auto_steps

the global steps of current task.




---

#### <kbd>property</kbd> global_train_epochs

the global train epochs of current task.




---

#### <kbd>property</kbd> global_train_steps

the global train steps of current task.




---

#### <kbd>property</kbd> grad_acc_steps

the grad accumulator steps of current task.




---

#### <kbd>property</kbd> grad_scaler

the grad scaler of current task.




---

#### <kbd>property</kbd> launcher

the current launcher.




---

#### <kbd>property</kbd> name

the node name in the current activated [`ExecutableGraph`](./core.graph.md#class-executablegraph).




---

#### <kbd>property</kbd> out_dir

the output directory of current task.




---

#### <kbd>property</kbd> run_id

the run id of current task.




---

#### <kbd>property</kbd> step_mode

whether current task is running by step (True) or by epoch (False).




---

#### <kbd>property</kbd> task

the current task.




---

#### <kbd>property</kbd> training

whether current task is training.






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/loss.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/loss.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_impl`

```python
forward_impl(cache: 'GraphOutputCache')
```








