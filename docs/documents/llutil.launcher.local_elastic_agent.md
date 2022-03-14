<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/local_elastic_agent.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.launcher.local_elastic_agent`






**Global Variables**
---------------
- **DEFAULT_ROLE**


---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/local_elastic_agent.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LocalElasticAgent`
An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
that handles host-local workers.
This agent is deployed per host and is configured to spawn `n` workers.
When using GPUs, `n` maps to the number of GPUs available on the host.


The local agent does not communicate to other local agents deployed on
other hosts, even if the workers may communicate inter-host. The worker id
is interpreted to be a local process. The agent starts and stops all worker
processes as a single unit.




The worker function and argument passed to the worker function must be
python multiprocessing compatible. To pass multiprocessing data structures
to the workers you may create the data structure in the same multiprocessing
context as the specified `start_method` and pass it as a function argument.


The `exit_barrier_timeout` specifies the amount of time (in seconds) to wait
for other agents to finish. This acts as a safety net to handle cases where
workers finish at different times, to prevent agents from viewing workers
that finished early as a scale-down event. It is strongly advised that the
user code deal with ensuring that workers are terminated in a synchronous
manner rather than relying on the exit_barrier_timeout.


Example launching function


:
```


     def trainer(args) -> str:
         return "do train"


     def main():
         start_method="spawn"
         shared_queue= multiprocessing.get_context(start_method).Queue()
         spec = WorkerSpec(
                     role="trainer",
                     local_world_size=nproc_per_process,
                     entrypoint=trainer,
                     args=("foobar",),
                     ...<OTHER_PARAMS...>)
         agent = LocalElasticAgent(spec, start_method)
         results = agent.run()


         if results.is_failed():
             print("trainer failed")
         else:
             print(f"rank 0 return value: {results.return_values[0]}")
             # prints -> rank 0 return value: do train


```
Example launching binary


:
```


     def main():
         spec = WorkerSpec(
                     role="trainer",
                     local_world_size=nproc_per_process,
                     entrypoint="/usr/local/bin/trainer",
                     args=("--trainer_args", "foobar"),
                     ...<OTHER_PARAMS...>)
         agent = LocalElasticAgent(spec)
         results = agent.run()


         if not results.is_failed():
             print("binary launches do not have return values")




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/launcher/local_elastic_agent.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    spec: WorkerSpec,
    start_method='spawn',
    exit_barrier_timeout: float = 300,
    log_dir: Optional[str] = None,
    events: Events = None
)
```










---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/llutil/launcher/local_elastic_agent/run#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(role: str = 'default') → RunResult
```








