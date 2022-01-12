# ice-learn
> `ice` is a modular high-level deep learning framework that extends and integrates PyTorch and PyCUDA with intuitive interfaces. We aims not only to minimize the boilerplate code without loss of functionality, but also maximize the flexibility and usability for extending and composing any deep learning tasks into an integrate multi-task learning program.


**NOTE:** `ice` is currently in pre-alpha versions, and the API is subject to change.

## Features 

- **Minimize Boilerplates**: 
    - **(Almost) Single API:** Modular arrangement of Neural Network training, test, and deploy utilities with a single main API class `ice.Assistant()`, with all the same pattern of methods `.add_...()`, e.g. `.add_dataset()`, `.add_model()`, etc. Basically you need to memorize no more function names because the IDE will then automatically promt the rest choices for you. Other functionalities not suitable for this main API will also keep things the same simple feeling such as using decorators to define new data pre-processing pipelines, which will all be well documented with demo codes.
    - **No extra Configuration File** You can compose miscellaneous modules with the main API in a single python script that serves both as the program entrance point and configuration file.
- **Maximize Flexiblility**: Painless and Incremental Extension from CUDA to non-standard data-preprocessing and training schedules for multi-task learning.
    - The kernel data structure of `ice` is a **Hypergraph** that manages different module nodes (e.g. `ice.DatasetNode`, `ice.ModuleNode`, etc.) that are switchable between multiple user-defined execution modes. Extending a new dataset, network module or loss function is by adding new `nn.Dataset`s, `nn.Module`s and python `callable`s to specific mode of the entire graph.
    - We provide **PyCUDA** support by automatically managing the PyCUDA context as well as providing a simplified `torch.Tensor` class wrapper that supports efficient multi-dimensional element access in CUDA codes. This feature manages to make writing, compile, execution and testing CUDA extensions for PyTorch extremely fast. We also provide a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=huangyuyao.pycuda-highlighter) for PyCUDA docstring highlight.
    - We support **Multi-Task Learning** training by finding the **Pareto Optimal** for each task weight so that you do not need to tune them manually.
    - We support **Lambda Function and Closure** for multiprocess applications. You can not only use built-in Data Distributed Parallel training without extra configurations, but also without constraints on not able to using inplace lambda functions (in the case of plain PyTorch). We provide many easy to use APIs based on this feature so that users can benefit from defining their data processing and network module input/output transform in a functional programming like paradigm.


## Install

`pip install ice-learn`

## Documentation

Please see the [docs](./docs) subdirectory.