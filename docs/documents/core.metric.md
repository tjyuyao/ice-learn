<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `core.metric`








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Meter`
value reducer that works recursively.







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(*args, **kwds)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(value, *args)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DictMetric`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(meters=None, meter_prototype=None)
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(*args, **kwds)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(explicit={}, *shared_args, **kwds)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MetricNode`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `better`

```python
better(new_value) → bool
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_end`

```python
epoch_end()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `epoch_start`

```python
epoch_start()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tensorboard_log_metric`

```python
tensorboard_log_metric(postfix='')
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValueMeter`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(value: Tensor)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SummationMeter`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(batch_sum: Tensor)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AverageMeter`









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(batch_avg: Tensor, count: int = 1)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MovingAverageMeter`






<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(window_size: int) → None
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/core/metric.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update`

```python
update(*values)
```








