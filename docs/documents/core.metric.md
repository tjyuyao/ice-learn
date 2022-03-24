<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.metric`






**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Meter`
value reducer that works recursively.







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(*args, **kwds)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(value, *args)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DictMetric`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(meters=None, meter_prototype=None)
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(*args, **kwds)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(explicit={}, *shared_args, **kwds)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MetricNode`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds)
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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `better`

```python
better(new_value) → bool
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_end`

```python
epoch_end()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_start`

```python
epoch_start()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tensorboard_log_metric`

```python
tensorboard_log_metric(postfix='')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValueMeter`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(value: 'Tensor')
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SummationMeter`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(batch_sum: 'Tensor')
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AverageMeter`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(batch_avg: 'Tensor', count: 'int' = 1)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MovingAverageMeter`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(window_size: 'int') → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(*values)
```








