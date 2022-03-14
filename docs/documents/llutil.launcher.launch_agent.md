<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/launch_agent.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.launcher.launch_agent`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/launch_agent.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `launch_agent`

```python
launch_agent(
    config: LaunchConfig,
    entrypoint: Optional[Callable, str],
    args: List[Any],
    ice_events: Events
) → Dict[int, Any]
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/launch_agent.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LaunchConfig`
Creates a rendezvous config.




**Args:**


 - <b>`min_nodes`</b>:  Minimum amount of nodes that the user function will
 be launched on. Elastic agent ensures that the user
 function start only when the min_nodes amount enters
 the rendezvous.

 - <b>`max_nodes`</b>:  Maximum amount of nodes that the user function
 will be launched on.

 - <b>`nproc_per_node`</b>:  On each node the elastic agent will launch
 this amount of workers that will execute user
 defined function.

 - <b>`rdzv_backend`</b>:  rdzv_backend to use in the rendezvous (zeus-adapter, etcd).

 - <b>`rdzv_endpoint`</b>:  The endpoint of the rdzv sync. storage.

 - <b>`rdzv_configs`</b>:  Key, value pair that specifies rendezvous specific configuration.

 - <b>`rdzv_timeout`</b>:  Legacy argument that specifies timeout for the rendezvous. It is going
 to be removed in future versions, see the note below. The default timeout is 900 seconds.

 - <b>`rdzv_id`</b>:  The unique run id of the job (if not passed a unique one will be
 deduced from run environment - flow workflow id in flow - or auto generated).

 - <b>`role`</b>:  User defined role of the worker (defaults to "trainer").

 - <b>`max_restarts`</b>:  The maximum amount of restarts that elastic agent will conduct
 on workers before failure.

 - <b>`monitor_interval`</b>:  The interval in seconds that is used by the elastic_agent
 as a period of monitoring workers.

 - <b>`start_method`</b>:  The method is used by the elastic agent to start the
 workers (spawn, fork, forkserver).

 - <b>`log_dir`</b>:  base log directory where log files are written. If not set,
 one is created in a tmp dir but NOT removed on exit.

 - <b>`redirects`</b>:  configuration to redirect stdout/stderr to log files.
 Pass a single `Std` enum to redirect all workers,
 or a mapping keyed by local_rank to selectively redirect.

 - <b>`tee`</b>:  configuration to "tee" stdout/stderr to console + log file.

 - <b>`metrics_cfg`</b>:  configuration to initialize metrics.


..note:
`rdzv_timeout` is a legacy argument that will be removed in future.
Set the timeout via `rdzv_configs['timeout']`




<a href="https://github.com/tjyuyao/ice-learn/blob/main/<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    min_nodes: int,
    max_nodes: int,
    nproc_per_node: int,
    run_id: str = '',
    role: str = 'default_role',
    rdzv_endpoint: str = '',
    rdzv_backend: str = 'etcd',
    rdzv_configs: Dict[str, Any] = <factory>,
    rdzv_timeout: int = -1,
    max_restarts: int = 3,
    monitor_interval: float = 30,
    start_method: str = 'spawn',
    log_dir: Optional[str] = None,
    redirects: Union[Std, Dict[int, Std]] = <Std.NONE: 0>,
    tee: Union[Std, Dict[int, Std]] = <Std.NONE: 0>,
    metrics_cfg: Dict[str, str] = <factory>
) → None
```











