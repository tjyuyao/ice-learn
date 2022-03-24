<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.module`






**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModuleNode`
a node that extends `torch.nn.Module`


`ModuleNode` manages neural network modules (`torch.nn.Module`) and the optimizers responsible to train them. For each `ModuleNode`, multiple optimizers can be specified, each of which can be responsible for different group of parameters by filtering parameters names.


Multiple `ModelNode` with different training configuration under differnt tags can share a same `torch.nn.Module`.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```








---

#### <kbd>property</kbd> board







---

#### <kbd>property</kbd> device

the assigned device by current launcher.




---

#### <kbd>property</kbd> epoch_size







---

#### <kbd>property</kbd> epoch_steps







---

#### <kbd>property</kbd> global_auto_steps







---

#### <kbd>property</kbd> global_train_epochs







---

#### <kbd>property</kbd> global_train_steps







---

#### <kbd>property</kbd> grad_acc_steps







---

#### <kbd>property</kbd> grad_scaler







---

#### <kbd>property</kbd> launcher







---

#### <kbd>property</kbd> name

the node name in the current activated [`ExecutableGraph`](./core.graph.md#class-executablegraph).




---

#### <kbd>property</kbd> out_dir







---

#### <kbd>property</kbd> run_id







---

#### <kbd>property</kbd> step_mode

whether current task is running by step (True) or by epoch (False).




---

#### <kbd>property</kbd> task







---

#### <kbd>property</kbd> training

whether current task is training.






---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_end`

```python
epoch_end()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_start`

```python
epoch_start()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_impl`

```python
forward_impl(cache: "'GraphOutputCache'")
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict: 'Dict', strict: 'bool')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/module.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update()
```








