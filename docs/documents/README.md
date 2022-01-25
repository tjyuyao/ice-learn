<!-- markdownlint-disable -->

# API Overview

## Modules

- [`api`](./api.md#module-api)
- [`api.scripts`](./api.scripts.md#module-apiscripts)
- [`api.scripts.waitfinish`](./api.scripts.waitfinish.md#module-apiscriptswaitfinish): usage: waitfinish [-h] PIDS [PIDS ...]
- [`core`](./core.md#module-core)
- [`core.dataset`](./core.dataset.md#module-coredataset)
- [`core.graph`](./core.graph.md#module-coregraph): An executable configuration graph.
- [`core.hypergraph`](./core.hypergraph.md#module-corehypergraph)
- [`core.loss`](./core.loss.md#module-coreloss)
- [`core.metric`](./core.metric.md#module-coremetric)
- [`core.module`](./core.module.md#module-coremodule)
- [`core.optim`](./core.optim.md#module-coreoptim)
- [`llutil`](./llutil.md#module-llutil)
- [`llutil.argparser`](./llutil.argparser.md#module-llutilargparser): This module provides helper functions for commonly used argument processing for functions, 
- [`llutil.collections`](./llutil.collections.md#module-llutilcollections)
- [`llutil.config`](./llutil.config.md#module-llutilconfig)
- [`llutil.dictprocess`](./llutil.dictprocess.md#module-llutildictprocess)
- [`llutil.launcher`](./llutil.launcher.md#module-llutillauncher)
- [`llutil.logging`](./llutil.logging.md#module-llutillogging)
- [`llutil.multiprocessing`](./llutil.multiprocessing.md#module-llutilmultiprocessing): ice.llutil.multiprocessing is a modified version of ``torch.multiprocessing``. It's designed to change
- [`llutil.multiprocessing.pool`](./llutil.multiprocessing.pool.md#module-llutilmultiprocessingpool)
- [`llutil.multiprocessing.queue`](./llutil.multiprocessing.queue.md#module-llutilmultiprocessingqueue)
- [`llutil.multiprocessing.reductions`](./llutil.multiprocessing.reductions.md#module-llutilmultiprocessingreductions)
- [`llutil.multiprocessing.spawn`](./llutil.multiprocessing.spawn.md#module-llutilmultiprocessingspawn)
- [`llutil.pycuda`](./llutil.pycuda.md#module-llutilpycuda)
- [`llutil.test`](./llutil.test.md#module-llutiltest)

## Classes

- [`graph.ExecutableGraph`](./core.graph.md#class-executablegraph)
- [`graph.InvalidURIError`](./core.graph.md#class-invalidurierror): An Exception raised when valid node URI is expected.
- [`graph.Node`](./core.graph.md#class-node)
- [`graph.NodeOutputCache`](./core.graph.md#class-nodeoutputcache)
- [`graph.StopTask`](./core.graph.md#class-stoptask): An Exception raised to exit current task.
- [`hypergraph.HyperGraph`](./core.hypergraph.md#class-hypergraph): HyperGraph is the container for all nodes.
- [`hypergraph.Repeat`](./core.hypergraph.md#class-repeat)
- [`hypergraph.Task`](./core.hypergraph.md#class-task)
- [`argparser.ArgumentMissingError`](./llutil.argparser.md#class-argumentmissingerror): Raised when a required argument is missing from command line.
- [`argparser.ArgumentTypeError`](./llutil.argparser.md#class-argumenttypeerror): Raised when converting an argument failed.
- [`argparser.FlexibleArgParser`](./llutil.argparser.md#class-flexibleargparser): A flexible and lightweight argument parser that saves loads of code.
- [`collections.ConfigDict`](./llutil.collections.md#class-configdict): stores multi-level configurations easily.
- [`collections.Counter`](./llutil.collections.md#class-counter): count values by group.
- [`collections.Dict`](./llutil.collections.md#class-dict): access dict values as attributes.
- [`config.Configurable`](./llutil.config.md#class-configurable)
- [`launcher.ElasticLauncher`](./llutil.launcher.md#class-elasticlauncher): **Example:**
- [`pool.Pool`](./llutil.multiprocessing.pool.md#class-pool): Pool implementation which uses our version of SimpleQueue.
- [`queue.ConnectionWrapper`](./llutil.multiprocessing.queue.md#class-connectionwrapper): Proxy class for _multiprocess.Connection which uses ForkingPickler to
- [`queue.Queue`](./llutil.multiprocessing.queue.md#class-queue)
- [`queue.SimpleQueue`](./llutil.multiprocessing.queue.md#class-simplequeue)
- [`reductions.SharedCache`](./llutil.multiprocessing.reductions.md#class-sharedcache): dictionary from multiprocess handles to StorageWeakRef
- [`reductions.StorageWeakRef`](./llutil.multiprocessing.reductions.md#class-storageweakref): A weak reference to a Storage.
- [`spawn.ProcessContext`](./llutil.multiprocessing.spawn.md#class-processcontext)
- [`spawn.ProcessException`](./llutil.multiprocessing.spawn.md#class-processexception)
- [`spawn.ProcessExitedException`](./llutil.multiprocessing.spawn.md#class-processexitedexception): Exception is thrown when the process failed due to signal
- [`spawn.ProcessRaisedException`](./llutil.multiprocessing.spawn.md#class-processraisedexception): Exception is thrown when the process failed due to exception
- [`spawn.SpawnContext`](./llutil.multiprocessing.spawn.md#class-spawncontext)
- [`pycuda.CUDAModule`](./llutil.pycuda.md#class-cudamodule): Just-In-Time compilation of a set of CUDA kernel functions and device functions from source.

## Functions

- [`argparser.as_dict`](./llutil.argparser.md#function-as_dict): Helper function: regularize input into a dict.
- [`argparser.as_list`](./llutil.argparser.md#function-as_list): Helper function: regularize input into list of element.
- [`argparser.isa`](./llutil.argparser.md#function-isa): Helper function: alias for python built-in ``isinstance``.
- [`config.clone`](./llutil.config.md#function-clone): clone configurables, containers, and ordinary objects recursively.
- [`config.configurable`](./llutil.config.md#function-configurable): This decorator delays the initialization of ``cls`` until ``freeze()``.
- [`config.freeze`](./llutil.config.md#function-freeze): freeze configurables recursively.
- [`config.has_builder`](./llutil.config.md#function-has_builder)
- [`config.is_configurable`](./llutil.config.md#function-is_configurable): check if a class or an object is configurable.
- [`config.make_configurable`](./llutil.config.md#function-make_configurable): This function converts multiple existing classes to configurables.
- [`config.objattr`](./llutil.config.md#function-objattr)
- [`dictprocess.Collect`](./llutil.dictprocess.md#function-collect): a predefined DictProcessor that keep only selected entries.
- [`dictprocess.Compose`](./llutil.dictprocess.md#function-compose): a predefined DictProcessor that composes a list of other DictProcessors together.
- [`dictprocess.dictprocess`](./llutil.dictprocess.md#function-dictprocess): ``ice.dictprocess`` is a function decorator that convert any function into a callable DictProcessor class that would take a dict as input and update its content.
- [`launcher.parse_devices_and_backend`](./llutil.launcher.md#function-parse_devices_and_backend)
- [`launcher.parse_min_max_nnodes`](./llutil.launcher.md#function-parse_min_max_nnodes)
- [`multiprocessing.called_from_main`](./llutil.multiprocessing.md#function-called_from_main): Another version of ``if __name__ == "__main__"`` that works everywhere.
- [`pool.clean_worker`](./llutil.multiprocessing.pool.md#function-clean_worker)
- [`reductions.fd_id`](./llutil.multiprocessing.reductions.md#function-fd_id)
- [`reductions.init_reductions`](./llutil.multiprocessing.reductions.md#function-init_reductions)
- [`reductions.rebuild_cuda_tensor`](./llutil.multiprocessing.reductions.md#function-rebuild_cuda_tensor)
- [`reductions.rebuild_event`](./llutil.multiprocessing.reductions.md#function-rebuild_event)
- [`reductions.rebuild_storage_empty`](./llutil.multiprocessing.reductions.md#function-rebuild_storage_empty)
- [`reductions.rebuild_storage_fd`](./llutil.multiprocessing.reductions.md#function-rebuild_storage_fd)
- [`reductions.rebuild_storage_filename`](./llutil.multiprocessing.reductions.md#function-rebuild_storage_filename)
- [`reductions.rebuild_tensor`](./llutil.multiprocessing.reductions.md#function-rebuild_tensor)
- [`reductions.reduce_event`](./llutil.multiprocessing.reductions.md#function-reduce_event)
- [`reductions.reduce_storage`](./llutil.multiprocessing.reductions.md#function-reduce_storage)
- [`reductions.reduce_tensor`](./llutil.multiprocessing.reductions.md#function-reduce_tensor)
- [`reductions.storage_from_cache`](./llutil.multiprocessing.reductions.md#function-storage_from_cache)
- [`spawn.spawn`](./llutil.multiprocessing.spawn.md#function-spawn): Spawns ``nprocs`` processes that run ``fn`` with ``args``.
- [`spawn.start_processes`](./llutil.multiprocessing.spawn.md#function-start_processes)
- [`test.requires_n_gpus`](./llutil.test.md#function-requires_n_gpus)
