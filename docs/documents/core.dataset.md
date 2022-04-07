<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.dataset`






**Global Variables**
---------------
- **string_classes**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `failsafe_collate`

```python
failsafe_collate(batch)
```

Puts each data field into a tensor with outer dimension batch size





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ResumableDistributedSampler`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset: Dataset,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
    num_iters: int = None
) → None
```








---

#### <kbd>property</kbd> indices









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_start_batch_idx`

```python
set_start_batch_idx(i)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DatasetNode`
Automating DataLoader and DataSampler creation and maintainance.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_impl`

```python
forward_impl(_)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_state_dict`

```python
load_state_dict(_state_dict: Dict, strict: bool = None)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/dataset.py#L249"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict() → Dict
```








