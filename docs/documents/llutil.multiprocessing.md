<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.multiprocessing`
a drop-in replacement for `torch.multiprocessing`.


ice.llutil.multiprocessing is a modified version of `torch.multiprocessing`. It's designed to change
``import torch.multiprocessing as mp`` to ``from ice import multiprocessing as mp`` to have all the lambda functions, 
closures as well as pytorch tensors sent through processes in Data Distributed Parallel paradigm.


Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.




**Global Variables**
---------------
- **reductions**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/__init__.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `called_from_main`

```python
called_from_main()
```

Another version of ``if __name__ == "__main__"`` that works everywhere.




**Returns:**

 bool





