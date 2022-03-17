<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/launcher.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.launcher.launcher`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/launcher.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_current_launcher`

```python
get_current_launcher() → ElasticLauncher
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/launcher.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ElasticLauncher`
A helper [`Configurable`](./llutil.config.md#class-configurable) class for `torchrun` and `torch.distributed.launch`.


PyTorch's elastic launch ability is embeded in this Configurable, for details please see [here](https://pytorch.org/docs/stable/elastic/run.html).


[`HyperGraph.run()`](./core.hypergraph.md#method-run) uses this class to launch multiple processes. Directly usage is also possible (see the example below).


**Example:**


```python
def worker(launcher):
     print("rank", launcher.rank)
     print("local_rank", launcher.local_rank)
     print("device", launcher.assigned_device)


if __name__ == "__main__":
     launcher = ElasticLauncher("cuda:*").freeze()
     launcher(worker, launcher)
```




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/launcher.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwds) → None
```








---

#### <kbd>property</kbd> assigned_device







---

#### <kbd>property</kbd> devices







---

#### <kbd>property</kbd> dist_backend







---

#### <kbd>property</kbd> group_rank







---

#### <kbd>property</kbd> group_world_size







---

#### <kbd>property</kbd> local_rank







---

#### <kbd>property</kbd> local_world_size







---

#### <kbd>property</kbd> master_addr







---

#### <kbd>property</kbd> master_port







---

#### <kbd>property</kbd> max_restarts







---

#### <kbd>property</kbd> rank







---

#### <kbd>property</kbd> rdzv_id







---

#### <kbd>property</kbd> restart_count







---

#### <kbd>property</kbd> role_name







---

#### <kbd>property</kbd> role_rank







---

#### <kbd>property</kbd> role_world_size







---

#### <kbd>property</kbd> world_size










