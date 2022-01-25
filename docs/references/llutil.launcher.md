<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.launcher`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_min_max_nnodes`

```python
parse_min_max_nnodes(nnodes: str)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_devices_and_backend`

```python
parse_devices_and_backend(devices: str, dist_backend: str)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ElasticLauncher`
**Example:**


```python
def run(args):
     local_rank = int(os.environ["LOCAL_RANK"])
     print(local_rank, args)


if __name__ == "__main__":
     launch = ElasticLauncher()
     launch['nproc_per_node'] = 2
     launch.freeze()
     launch(run, "blabla")
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










