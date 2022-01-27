<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.launcher`








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__freeze__`

```python
__freeze__(
    devices='auto',
    nnodes='1:1',
    dist_backend='auto',
    rdzv_id='none',
    rdzv_endpoint='',
    rdzv_backend='static',
    rdzv_configs='',
    standalone=False,
    max_restarts=0,
    monitor_interval=5,
    start_method='spawn',
    redirects='0',
    tee='0',
    log_dir=None,
    role='default',
    node_rank=0,
    master_addr='127.0.0.1',
    master_port=None,
    omp_num_threads=1
)
```

**Args:**

- **Worker/node size related arguments:**

    - **`devices`** (str, optional): enumerates devices on this node, e.g.: [`"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:*"`, `"auto:*"`, `"cuda:1,3"`, `"cuda:0-2,7"`]. Defaults to `"auto"`.

    - **`dist_backend`** (str, optional): supports: [`"nccl"`, `"gloo"`, `"mpi"`, `"auto"`]. If given `"auto"`, will use `"nccl"` for `"cuda"` and `"gloo"` for `"cpu"` in general. Defaults to `"auto"`.

    - **`nnodes`** (str, optional): Number of nodes, or the range of nodes in form `<minimum_nodes>:<maximum_nodes>`
 . Defaults to `"1:1"`.

- **Rendezvous related arguments:**

    - **`rdzv_id`** (str, optional): User-defined group id.

    - **`rdzv_endpoint`** (str, optional): Rendezvous backend endpoint; usually in form `<host>:<port>`.

    - **`rdzv_backend`** (str, optional): Rendezvous backend.

    - **`rdzv_configs`** (str, optional): Additional rendezvous configuration (`<key1>=<value1>,<key2>=<value2>,...`).

    - **`standalone`** (bool, optional): Start a local standalone rendezvous backend that is represented by a C10d TCP store on port 29400. Useful when launching single-node, multi-worker job. If specified rdzv_backend, rdzv_endpoint, rdzv_id are auto-assigned; any explicitly set values are ignored. Defaults to `False`.

- **User-code launch related arguments:**

    - **`max_restarts`** (int, optional): Maximum number of worker group restarts before failing. Defaults to 0.

    - **`monitor_interval`** (int, optional): Interval, in seconds, to monitor the state of workers. Defaults to 5.

    - **`start_method`** (str, optional): Multiprocessing start method to use when creating workers. Defaults to `"spawn"`.

    - **`redirects`** (str, optional): Redirect std streams into a log file in the log directory (e.g. `3` redirects both stdout+stderr for all workers, `0:1,1:2` redirects stdout for local rank 0 and stderr for local rank 1). Defaults to `"0"`.

    - **`tee`** (str, optional): Tee std streams into a log file and also to console (see redirects for format). Defaults to `"0"`.

    - **`log_dir`** ([type], optional): Base directory to use for log files (e.g. /var/log/torch/elastic). The same directory is re-used for multiple runs (a unique job-level sub-directory is created with rdzv_id as the prefix). Defaults to None.

    - **`role`** (str, optional): User-defined role for the workers. Defaults to `"default"`.

- **Backwards compatible parameters with `caffe2.distributed.launch`:**

    - **`node_rank`** (int, optional): "Rank of the node for multi-node distributed training."). Defaults to 0.

    - **`master_addr`** (str, optional): Address of the master node (rank 0). It should be either the IP address or the hostname of rank 0. For single node multi-proc training the master_addr can simply be 127.0.0.1; IPv6 should have the pattern `[0:0:0:0:0:0:0:1]`.") Defaults to "127.0.0.1".

    - **`master_port`** ([type], optional): Port on the master node (rank 0) to be used for communication during distributed training. Defaults will generate a random port between `16894` and `17194`.

    - **`omp_num_threads`** (int, optional): set `OMP_NUM_THREADS` environment if not exists. Defaults to 1.





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










