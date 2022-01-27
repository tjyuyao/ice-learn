<!-- markdownlint-disable -->

# API Overview

## Modules

- [`api.scripts.wait_process`](./api.scripts.wait_process.md#module-apiscriptswait_process): wait for a process to finish.
- [`core.graph`](./core.graph.md#module-coregraph): contains [`Node`](./core.graph.md#class-node) and [`ExecutableGraph`](./core.graph.md#class-executablegraph).
- [`core.hypergraph`](./core.hypergraph.md#module-corehypergraph)
- [`llutil`](./llutil.md#module-llutil)
- [`llutil.argparser`](./llutil.argparser.md#module-llutilargparser): parse arguments for functions and command line.
- [`llutil.collections`](./llutil.collections.md#module-llutilcollections)
- [`llutil.config`](./llutil.config.md#module-llutilconfig)
- [`llutil.dictprocess`](./llutil.dictprocess.md#module-llutildictprocess)
- [`llutil.launcher`](./llutil.launcher.md#module-llutillauncher)
- [`llutil.logging`](./llutil.logging.md#module-llutillogging): logging utilities.
- [`llutil.multiprocessing`](./llutil.multiprocessing.md#module-llutilmultiprocessing): a drop-in replacement for `torch.multiprocessing`.
- [`llutil.pycuda`](./llutil.pycuda.md#module-llutilpycuda): Integrates PyCUDA to PyTorch and ice.
- [`llutil.test`](./llutil.test.md#module-llutiltest): helps developers of ice to test.

## Classes

- [`graph.ExecutableGraph`](./core.graph.md#class-executablegraph)
- [`graph.GraphOutputCache`](./core.graph.md#class-graphoutputcache)
- [`graph.InvalidURIError`](./core.graph.md#class-invalidurierror): An Exception raised when valid node URI is expected.
- [`graph.Node`](./core.graph.md#class-node): This class defines the executable node.
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
- [`launcher.ElasticLauncher`](./llutil.launcher.md#class-elasticlauncher): A helper [`Configurable`](./llutil.config.md#class-configurable) class for `torchrun` and `torch.distributed.launch`.
- [`pycuda.CUDAModule`](./llutil.pycuda.md#class-cudamodule): Just-In-Time compilation of a set of CUDA kernel functions and device functions from source.

## Functions

- [`argparser.as_dict`](./llutil.argparser.md#function-as_dict): helps to regularize input into a dict.
- [`argparser.as_list`](./llutil.argparser.md#function-as_list): helps to regularize input into list of element.
- [`argparser.isa`](./llutil.argparser.md#function-isa): an alias for python built-in `isinstance`.
- [`config.clone`](./llutil.config.md#function-clone): clone configurables, containers, and ordinary objects recursively.
- [`config.configurable`](./llutil.config.md#function-configurable): This decorator delays the initialization of `cls` until [`freeze()`](./llutil.config.md#function-freeze).
- [`config.freeze`](./llutil.config.md#function-freeze): freeze configurables recursively.
- [`config.has_builder`](./llutil.config.md#function-has_builder)
- [`config.is_configurable`](./llutil.config.md#function-is_configurable): check if a class or an object is configurable.
- [`config.make_configurable`](./llutil.config.md#function-make_configurable): This function converts multiple existing classes to configurables.
- [`config.objattr`](./llutil.config.md#function-objattr)
- [`dictprocess.Collect`](./llutil.dictprocess.md#function-collect): a predefined DictProcessor that keep only selected entries.
- [`dictprocess.Compose`](./llutil.dictprocess.md#function-compose): a predefined DictProcessor that composes a list of other DictProcessors together.
- [`dictprocess.dictprocess`](./llutil.dictprocess.md#function-dictprocess): a decorator that convert function into a DictProcessor (`Callable[[Dict], Dict]`).
- [`logging.get_logger`](./llutil.logging.md#function-get_logger): set up a simple logger that writes into stderr. 
- [`multiprocessing.called_from_main`](./llutil.multiprocessing.md#function-called_from_main): Another version of ``if __name__ == "__main__"`` that works everywhere.
- [`test.requires_n_gpus`](./llutil.test.md#function-requires_n_gpus)
