<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.multiprocessing.reductions`






**Global Variables**
---------------
- **shared_cache**

---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rebuild_event`

```python
rebuild_event(device, handle)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reduce_event`

```python
reduce_event(event)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rebuild_tensor`

```python
rebuild_tensor(cls, storage, metadata)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rebuild_cuda_tensor`

```python
rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required
)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reduce_tensor`

```python
reduce_tensor(tensor)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L305"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fd_id`

```python
fd_id(fd)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `storage_from_cache`

```python
storage_from_cache(cls, key)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rebuild_storage_fd`

```python
rebuild_storage_fd(cls, df, size)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rebuild_storage_filename`

```python
rebuild_storage_filename(cls, manager, handle, size)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L342"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rebuild_storage_empty`

```python
rebuild_storage_empty(cls)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L346"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reduce_storage`

```python
reduce_storage(storage)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L373"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `init_reductions`

```python
init_reductions()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StorageWeakRef`
A weak reference to a Storage.


The cdata member is a Python number containing the integer representation of
the Storage pointer.







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `expired`

```python
expired()
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SharedCache`
dictionary from multiprocess handles to StorageWeakRef







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `free_dead_references`

```python
free_dead_references()
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/multiprocessing/reductions.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(key)
```








