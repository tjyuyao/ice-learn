<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.multiprocessing`
ice.llutil.multiprocessing is a rewrite of :mod:torch.multiprocessing. It's designed to change ``import torch.multiprocessing`` to ``import ice.multiprocessing`` to have all the lambda functions,  closures as well as pytorch tensors sent through processes in Data Distributed Parallel paradigm. 

Because of the similarity of APIs we do not document most of this package contents, and we recommend referring to very good docs of the original module. 



**Global Variables**
---------------
- **reductions**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/__init__.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_sharing_strategy`

```python
set_sharing_strategy(new_strategy)
```

Sets the strategy for sharing CPU tensors. 



**Args:**
 
 - <b>`new_strategy`</b> (str):  Name of the selected strategy. Should be one of 
 - <b>`the values returned by `</b>: func:`get_all_sharing_strategies()`. 




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/__init__.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sharing_strategy`

```python
get_sharing_strategy()
```

Returns the current strategy for sharing CPU tensors. 




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/__init__.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_all_sharing_strategies`

```python
get_all_sharing_strategies()
```

Returns a set of sharing strategies supported on a current system. 




