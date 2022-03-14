# ice-learn

<img src="https://s3.bmp.ovh/imgs/2022/03/89472f58c56b474c.jpg" alt="icon" width="250" style="display:block;margin-left:auto; margin-right:auto;" />

> `ice` is a sweet extension of PyTorch, a modular high-level deep learning framework that extends and integrates PyTorch and PyCUDA with intuitive interfaces. We aims not only to minimize the boilerplate code without loss of functionality, but also maximize the flexibility and usability for extending and composing any deep learning tasks into an integrate multi-task learning program.

**NOTE:** `ice` is currently in pre-alpha versions, and the API is subject to change.

## Features

- **Minimize Boilerplates**: You don't need to repeat yourself.
  - **Config Once, Use Everywhere:** Every mutable class can be converted into a `configurable`. Configuration for deep learning project has never been this easy before. A tagging system to manage and reuse any type of resources you need.
  - **Inplace Argument Parser:** You can parse command line argument instantly without a long page of previous definition.

- **Maximize Flexiblility**: Painless and Incremental Extension from CUDA to non-standard data-preprocessing and training schedules for multi-task learning.
  - The kernel data structure of `ice` is a **Hypergraph** that manages different module nodes (e.g. `ice.DatasetNode`, `ice.ModuleNode`, etc.) that are switchable between multiple user-defined execution modes. Extending a new dataset, network module or loss function is by adding new `nn.Dataset`s, `nn.Module`s and python `callable`s to specific mode of the entire graph.
  - We provide **PyCUDA** support by automatically managing the PyCUDA context as well as providing a simplified `torch.Tensor` class wrapper that supports efficient multi-dimensional element access in CUDA codes. This feature manages to make writing, compile, execution and testing CUDA extensions for PyTorch extremely fast. We also provide a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=huangyuyao.pycuda-highlighter) for PyCUDA docstring highlight.
  - We support **Multi-Task Learning** training by finding the **Pareto Optimal** for each task weight so that you do not need to tune them manually. (**TODO**)
  - We support [dill](https://github.com/uqfoundation/dill)-backended **Elastic Multiprocessing** launch and management that is compitable with **Lambda Function and Closures**. You can not only build multi-gpu or multi-machine Data Distributed Parallel training program without effort, but also doesn't require to concern about pickability of any part of program in your application. We actually suggest heavy use of lambda functions such as for simple input and output transforms of modules. This feature also contributes to the *minimal boilerplates* aim of `ice`.

## Install

`pip install ice-learn` **(TODO)**

**Note:** Currently installing via pip is not available, developers can install ice using poetry environment as specified [here](https://github.com/tjyuyao/ice-learn/blob/main/docs/resources/dev_notes/00_setup_devenv.md).

## Documentation

Please see the [docs](https://github.com/tjyuyao/ice-learn/tree/main/docs) subdirectory, where contains further informations about [Get Started Tutorials](https://github.com/tjyuyao/ice-learn/tree/main/docs/tutorials), [API Reference Manual](https://github.com/tjyuyao/ice-learn/tree/main/docs/references) and [Dev. Notes for Contributors](https://github.com/tjyuyao/ice-learn/tree/main/docs/devnotes).
