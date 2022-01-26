<!-- markdownlint-disable -->

# API Overview

## Modules

- [`api.scripts.wait_process`](./api.scripts.wait_process.md#module-apiscriptswait_process): wait for a process to finish.
- [`core.graph`](./core.graph.md#module-coregraph)
- [`core.hypergraph`](./core.hypergraph.md#module-corehypergraph)
- [`llutil.argparser`](./llutil.argparser.md#module-llutilargparser): This module provides helper functions for commonly used argument processing for functions, 
- [`llutil.collections`](./llutil.collections.md#module-llutilcollections)
- [`llutil.config`](./llutil.config.md#module-llutilconfig)
- [`llutil.dictprocess`](./llutil.dictprocess.md#module-llutildictprocess)
- [`llutil.launcher`](./llutil.launcher.md#module-llutillauncher)
- [`llutil.multiprocessing`](./llutil.multiprocessing.md#module-llutilmultiprocessing): ice.llutil.multiprocessing is a modified version of ``torch.multiprocessing``. It's designed to change
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
- [`pycuda.CUDAModule`](./llutil.pycuda.md#class-cudamodule): Just-In-Time compilation of a set of CUDA kernel functions and device functions from source.

## Functions

- [`argparser.as_dict`](./llutil.argparser.md#function-as_dict): Helper function: regularize input into a dict.
- [`argparser.as_list`](./llutil.argparser.md#function-as_list): Helper function: regularize input into list of element.
- [`argparser.isa`](./llutil.argparser.md#function-isa): Helper function: alias for python built-in `isinstance`.
- [`config.clone`](./llutil.config.md#function-clone): clone configurables, containers, and ordinary objects recursively.
- [`config.configurable`](./llutil.config.md#function-configurable): This decorator delays the initialization of `cls` until ``freeze()``.
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
- [`test.requires_n_gpus`](./llutil.test.md#function-requires_n_gpus)
