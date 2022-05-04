# ice-learn

<img align="left" width="128" height="128" src="https://s3.bmp.ovh/imgs/2022/03/e33e7e297b95b74c.jpg">

> `ice` is a sweet extension of PyTorch, a modular high-level deep learning framework that extends and integrates PyTorch and PyCUDA with intuitive interfaces. We aims not only to minimize the boilerplate code without loss of functionality, but also maximize the flexibility and usability for extending and composing any deep learning tasks into an integrate multi-task learning program.

**NOTE:** It is currently in pre-alpha versions, and the API is subject to change.

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

`pip install ice-learn` **(Recommended)**

or `pip install .[dev]` after a git-clone for developers.

## Documentation

You can access documentation through [Online Documentation Site](https://tjyuyao.github.io/ice-learn/), or the [`docs` subdirectory](https://github.com/tjyuyao/ice-learn/tree/main/docs) directly. The documentation is partial auto-generated from comment, and partial manually written, the note on how we produce the documenation can be found [here](https://tjyuyao.github.io/ice-learn/resources/dev_notes/02_docs_and_tests/).
