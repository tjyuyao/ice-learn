<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.file_client`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mmcv_mkdir_or_exist`

```python
mmcv_mkdir_or_exist(dir_name, mode=511)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `has_method`

```python
has_method(obj: object, method: str) → bool
```

Check whether the object has a method.




**Args:**


 - <b>`method`</b> (str):  The method name to check.

 - <b>`obj`</b> (object):  The object to check.




**Returns:**


 - <b>`bool`</b>:  True if the object has the method else False.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseStorageBackend`
Abstract class of storage backends.


All backends need to implement two apis: `get()` and `get_text()`.
`get()` reads the file as a byte stream and `get_text()` reads the file
as texts.





---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CephBackend`
Ceph storage backend (for internal use).




**Args:**


 - <b>`path_mapping`</b> (dict|None):  path mapping dict from local path to Petrel

 - <b>`path. When ``path_mapping={'src'`</b>:  'dst'}``, `src` in `filepath`

 - <b>`will be replaced by `dst`. Default`</b>:  None.


.. warning:
```
    :class:`mmcv.fileio.file_client.CephBackend` will be deprecated,
    please use :class:`mmcv.fileio.file_client.PetrelBackend` instead.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_mapping=None)
```








---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath, encoding=None)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PetrelBackend`
Petrel storage backend (for internal use).


PetrelBackend supports reading and writing data to multiple clusters.
If the file path contains the cluster name, PetrelBackend will read data
from specified cluster or write data to it. Otherwise, PetrelBackend will
access the default cluster.




**Args:**


 - <b>`path_mapping`</b> (dict, optional):  Path mapping dict from local path to

 - <b>`Petrel path. When ``path_mapping={'src'`</b>:  'dst'}``, `src` in

 - <b>``filepath` will be replaced by `dst`. Default`</b>:  None.

 - <b>`enable_mc`</b> (bool, optional):  Whether to enable memcached support.

 - <b>`Default`</b>:  True.




**Examples:**

```python
filepath1 = 's3://path/of/file'
    filepath2 = 'cluster-name:s3://path/of/file'
    client = PetrelBackend()
    client.get(filepath1)  # get data from default cluster
    client.get(filepath2)  # get data from 'cluster-name' cluster
```



<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path_mapping: Optional[dict] = None, enable_mc: bool = True)
```








---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `exists`

```python
exists(filepath: Union[str, Path]) → bool
```

Check whether a file path exists.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether exists.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` exists, `False` otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath: Union[str, Path]) → memoryview
```

Read data from a given `filepath` with 'rb' mode.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to read data.




**Returns:**


 - <b>`memoryview`</b>:  A memory view of expected bytes object to avoid
 copying. The memoryview object can be converted to bytes by
 `value_buf.tobytes()`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/llutil/file_client/get_local_path#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_local_path`

```python
get_local_path(filepath: Union[str, Path]) → Iterable[str]
```

Download a file from `filepath` and return a temporary path.


`get_local_path` is decorated by :meth:`contxtlib.contextmanager`. It
can be called with `with` statement, and when exists from the
`with` statement, the temporary path will be released.




**Args:**


 - <b>`filepath`</b> (str | Path):  Download a file from `filepath`.




**Examples:**

```python
client = PetrelBackend()
    # After existing from the `with` clause,
    # the path will be removed
    with client.get_local_path('s3://path/of/your/file') as path:
#     ...     # do something here
```



**Yields:**


 - <b>`Iterable[str]`</b>:  Only yield one temporary path.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath: Union[str, Path], encoding: str = 'utf-8') → str
```

Read data from a given `filepath` with 'r' mode.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to read data.

 - <b>`encoding`</b> (str):  The encoding format used to open the `filepath`.

 - <b>`Default`</b>:  'utf-8'.




**Returns:**


 - <b>`str`</b>:  Expected text reading from `filepath`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `isdir`

```python
isdir(filepath: Union[str, Path]) → bool
```

Check whether a file path is a directory.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether it is a
 directory.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` points to a directory,
`False` otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `isfile`

```python
isfile(filepath: Union[str, Path]) → bool
```

Check whether a file path is a file.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether it is a file.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` points to a file, `False`
otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `join_path`

```python
join_path(filepath: Union[str, Path], *filepaths: Union[str, Path]) → str
```

Concatenate all file paths.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be concatenated.




**Returns:**


 - <b>`str`</b>:  The result after concatenation.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `list_dir_or_file`

```python
list_dir_or_file(
    dir_path: Union[str, Path],
    list_dir: bool = True,
    list_file: bool = True,
    suffix: Optional[str, Tuple[str]] = None,
    recursive: bool = False
) → Iterator[str]
```

Scan a directory to find the interested directories or files in
arbitrary order.




**Note:**

> Petrel has no concept of directories but it simulates the directory
 hierarchy in the filesystem through public prefixes. In addition,
 if the returned path ends with '/', it means the path is a public
 prefix which is a logical directory.




**Note:**

> :meth:`list_dir_or_file` returns the path relative to `dir_path`.
 In addition, the returned path of directory will not contains the
 suffix '/' which is consistent with other backends.




**Args:**


 - <b>`dir_path`</b> (str | Path):  Path of the directory.

 - <b>`list_dir`</b> (bool):  List the directories. Default: True.

 - <b>`list_file`</b> (bool):  List the path of files. Default: True.

 - <b>`suffix`</b> (str or tuple[str], optional):   File suffix

 - <b>`that we are interested in. Default`</b>:  None.

 - <b>`recursive`</b> (bool):  If set to True, recursively scan the

 - <b>`directory. Default`</b>:  False.




**Yields:**


 - <b>`Iterable[str]`</b>:  A relative path to `dir_path`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `put`

```python
put(obj: bytes, filepath: Union[str, Path]) → None
```

Save data to a given `filepath`.




**Args:**


 - <b>`obj`</b> (bytes):  Data to be saved.

 - <b>`filepath`</b> (str or Path):  Path to write data.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `put_text`

```python
put_text(obj: str, filepath: Union[str, Path], encoding: str = 'utf-8') → None
```

Save data to a given `filepath`.




**Args:**


 - <b>`obj`</b> (str):  Data to be written.

 - <b>`filepath`</b> (str or Path):  Path to write data.

 - <b>`encoding`</b> (str):  The encoding format used to encode the `obj`.

 - <b>`Default`</b>:  'utf-8'.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove`

```python
remove(filepath: Union[str, Path]) → None
```

Remove a file.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be removed.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MemcachedBackend`
Memcached storage backend.




**Attributes:**


 - <b>`server_list_cfg`</b> (str):  Config file for memcached server list.

 - <b>`client_cfg`</b> (str):  Config file for memcached client.

 - <b>`sys_path`</b> (str | None):  Additional path to be appended to `sys.path`.

 - <b>`Default`</b>:  None.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L439"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(server_list_cfg, client_cfg, sys_path=None)
```








---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L463"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath, encoding=None)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L467"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LmdbBackend`
Lmdb storage backend.




**Args:**


 - <b>`db_path`</b> (str):  Lmdb database path.

 - <b>`readonly`</b> (bool, optional):  Lmdb environment parameter. If True,

 - <b>`disallow any write operations. Default`</b>:  True.

 - <b>`lock`</b> (bool, optional):  Lmdb environment parameter. If False, when

 - <b>`concurrent access occurs, do not lock the database. Default`</b>:  False.

 - <b>`readahead`</b> (bool, optional):  Lmdb environment parameter. If False,
 disable the OS filesystem readahead mechanism, which may improve
 random read performance when a database is larger than RAM.

 - <b>`Default`</b>:  False.




**Attributes:**


 - <b>`db_path`</b> (str):  Lmdb database path.




<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L485"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(db_path, readonly=True, lock=False, readahead=False, **kwargs)
```








---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L504"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath)
```

Get values according to the filepath.




**Args:**


 - <b>`filepath`</b> (str | obj:`Path`):  Here, filepath is the lmdb key.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L515"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath, encoding=None)
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L519"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HardDiskBackend`
Raw hard disks storage backend.





---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L597"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `exists`

```python
exists(filepath: Union[str, Path]) → bool
```

Check whether a file path exists.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether exists.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` exists, `False` otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L524"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath: Union[str, Path]) → bytes
```

Read data from a given `filepath` with 'rb' mode.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to read data.




**Returns:**


 - <b>`bytes`</b>:  Expected bytes object.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/llutil/file_client/get_local_path#L648"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_local_path`

```python
get_local_path(filepath: Union[str, Path]) → Iterable[Union[str, Path]]
```

Only for unified API and do nothing.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L537"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath: Union[str, Path], encoding: str = 'utf-8') → str
```

Read data from a given `filepath` with 'r' mode.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to read data.

 - <b>`encoding`</b> (str):  The encoding format used to open the `filepath`.

 - <b>`Default`</b>:  'utf-8'.




**Returns:**


 - <b>`str`</b>:  Expected text reading from `filepath`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L608"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `isdir`

```python
isdir(filepath: Union[str, Path]) → bool
```

Check whether a file path is a directory.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether it is a
 directory.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` points to a directory,
`False` otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L621"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `isfile`

```python
isfile(filepath: Union[str, Path]) → bool
```

Check whether a file path is a file.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether it is a file.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` points to a file, `False`
otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L633"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `join_path`

```python
join_path(filepath: Union[str, Path], *filepaths: Union[str, Path]) → str
```

Concatenate all file paths.


Join one or more filepath components intelligently. The return value
is the concatenation of filepath and any members of *filepaths.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be concatenated.




**Returns:**


 - <b>`str`</b>:  The result of concatenation.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L654"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `list_dir_or_file`

```python
list_dir_or_file(
    dir_path: Union[str, Path],
    list_dir: bool = True,
    list_file: bool = True,
    suffix: Optional[str, Tuple[str]] = None,
    recursive: bool = False
) → Iterator[str]
```

Scan a directory to find the interested directories or files in
arbitrary order.




**Note:**

> :meth:`list_dir_or_file` returns the path relative to `dir_path`.




**Args:**


 - <b>`dir_path`</b> (str | Path):  Path of the directory.

 - <b>`list_dir`</b> (bool):  List the directories. Default: True.

 - <b>`list_file`</b> (bool):  List the path of files. Default: True.

 - <b>`suffix`</b> (str or tuple[str], optional):   File suffix

 - <b>`that we are interested in. Default`</b>:  None.

 - <b>`recursive`</b> (bool):  If set to True, recursively scan the

 - <b>`directory. Default`</b>:  False.




**Yields:**


 - <b>`Iterable[str]`</b>:  A relative path to `dir_path`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `put`

```python
put(obj: bytes, filepath: Union[str, Path]) → None
```

Write data to a given `filepath` with 'wb' mode.




**Note:**

> `put` will create a directory if the directory of `filepath`
 does not exist.




**Args:**


 - <b>`obj`</b> (bytes):  Data to be written.

 - <b>`filepath`</b> (str or Path):  Path to write data.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L569"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `put_text`

```python
put_text(obj: str, filepath: Union[str, Path], encoding: str = 'utf-8') → None
```

Write data to a given `filepath` with 'w' mode.




**Note:**

> `put_text` will create a directory if the directory of
 `filepath` does not exist.




**Args:**


 - <b>`obj`</b> (str):  Data to be written.

 - <b>`filepath`</b> (str or Path):  Path to write data.

 - <b>`encoding`</b> (str):  The encoding format used to open the `filepath`.

 - <b>`Default`</b>:  'utf-8'.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L589"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove`

```python
remove(filepath: Union[str, Path]) → None
```

Remove a file.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be removed.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L707"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HTTPBackend`
HTTP and HTTPS storage bachend.





---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L710"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath)
```







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/llutil/file_client/get_local_path#L718"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_local_path`

```python
get_local_path(filepath: str) → Iterable[str]
```

Download a file from `filepath`.


`get_local_path` is decorated by :meth:`contxtlib.contextmanager`. It
can be called with `with` statement, and when exists from the
`with` statement, the temporary path will be released.




**Args:**


 - <b>`filepath`</b> (str):  Download a file from `filepath`.




**Examples:**

```python
client = HTTPBackend()
    # After existing from the `with` clause,
    # the path will be removed
    with client.get_local_path('http://path/of/your/file') as path:
#     ...     # do something here
```



---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L714"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath, encoding='utf-8')
```








---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L745"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FileClient`
A general file client to access files in different backends.


The client loads a file or text in a specified backend from its path
and returns it as a binary or text file. There are two ways to choose a
backend, the name of backend and the prefix of path. Although both of them
can be used to choose a storage backend, `backend` has a higher priority
that is if they are all set, the storage backend will be chosen by the
backend argument. If they are all `None`, the disk backend will be chosen.
Note that It can also register other backend accessor with a given name,
prefixes, and backend class. In addition, We use the singleton pattern to
avoid repeated object creation. If the arguments are the same, the same
object will be returned.




**Args:**


 - <b>`backend`</b> (str, optional):  The storage backend type. Options are "disk",

 - <b>`"ceph", "memcached", "lmdb", "http" and "petrel". Default`</b>:  None.

 - <b>`prefix`</b> (str, optional):  The prefix of the registered storage backend.

 - <b>`Options are "s3", "http", "https". Default`</b>:  None.




**Examples:**

```python
# only set backend
    file_client = FileClient(backend='petrel')
    # only set prefix
    file_client = FileClient(prefix='s3')
    # set both backend and prefix but use backend to choose client
    file_client = FileClient(backend='petrel', prefix='s3')
    # if the arguments are the same, the same object is returned
    file_client1 = FileClient(backend='petrel')
    file_client1 is file_client
#     True
```



**Attributes:**


 - <b>`client`</b> (:obj:`BaseStorageBackend`):  The backend object.





---

#### <kbd>property</kbd> allow_symlink







---

#### <kbd>property</kbd> name









---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1058"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `exists`

```python
exists(filepath: Union[str, Path]) → bool
```

Check whether a file path exists.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether exists.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` exists, `False` otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L991"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get`

```python
get(filepath: Union[str, Path]) → Union[bytes, memoryview]
```

Read data from a given `filepath` with 'rb' mode.




**Note:**

> There are two types of return values for `get`, one is `bytes`
 and the other is `memoryview`. The advantage of using memoryview
 is that you can avoid copying, and if you want to convert it to
 `bytes`, you can use ``.tobytes()``.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to read data.




**Returns:**


 - <b>`bytes | memoryview`</b>:  Expected bytes object or a memory view of the
bytes object.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/llutil/file_client/get_local_path#L1109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_local_path`

```python
get_local_path(filepath: Union[str, Path]) → Iterable[str]
```

Download data from `filepath` and write the data to local path.


`get_local_path` is decorated by :meth:`contxtlib.contextmanager`. It
can be called with `with` statement, and when exists from the
`with` statement, the temporary path will be released.




**Note:**

> If the `filepath` is a local path, just return itself.


.. warning:
```
     `get_local_path` is an experimental interface that may change in
     the future.


```


**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be read data.




**Examples:**

```python
file_client = FileClient(prefix='s3')
    with file_client.get_local_path('s3://bucket/abc.jpg') as path:
#     ...     # do something here
```



**Yields:**


 - <b>`Iterable[str]`</b>:  Only yield one path.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1009"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_text`

```python
get_text(filepath: Union[str, Path], encoding='utf-8') → str
```

Read data from a given `filepath` with 'r' mode.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to read data.

 - <b>`encoding`</b> (str):  The encoding format used to open the `filepath`.

 - <b>`Default`</b>:  'utf-8'.




**Returns:**


 - <b>`str`</b>:  Expected text reading from `filepath`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L872"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `infer_client`

```python
infer_client(
    file_client_args: Optional[dict] = None,
    uri: Optional[str, Path] = None
) → FileClient
```

Infer a suitable file client based on the URI and arguments.




**Args:**


 - <b>`file_client_args`</b> (dict, optional):  Arguments to instantiate a

 - <b>`FileClient. Default`</b>:  None.

 - <b>`uri`</b> (str | Path, optional):  Uri to be parsed that contains the file

 - <b>`prefix. Default`</b>:  None.




**Examples:**

```python
uri = 's3://path/of/your/file'
    file_client = FileClient.infer_client(uri=uri)
    file_client_args = {'backend': 'petrel'}
    file_client = FileClient.infer_client(file_client_args)
```



**Returns:**


 - <b>`FileClient`</b>:  Instantiated FileClient object.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1069"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `isdir`

```python
isdir(filepath: Union[str, Path]) → bool
```

Check whether a file path is a directory.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether it is a
 directory.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` points to a directory,
`False` otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1082"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `isfile`

```python
isfile(filepath: Union[str, Path]) → bool
```

Check whether a file path is a file.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be checked whether it is a file.




**Returns:**


 - <b>`bool`</b>:  Return `True` if `filepath` points to a file, `False`
otherwise.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1094"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `join_path`

```python
join_path(filepath: Union[str, Path], *filepaths: Union[str, Path]) → str
```

Concatenate all file paths.


Join one or more filepath components intelligently. The return value
is the concatenation of filepath and any members of *filepaths.




**Args:**


 - <b>`filepath`</b> (str or Path):  Path to be concatenated.




**Returns:**


 - <b>`str`</b>:  The result of concatenation.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `list_dir_or_file`

```python
list_dir_or_file(
    dir_path: Union[str, Path],
    list_dir: bool = True,
    list_file: bool = True,
    suffix: Optional[str, Tuple[str]] = None,
    recursive: bool = False
) → Iterator[str]
```

Scan a directory to find the interested directories or files in
arbitrary order.




**Note:**

> :meth:`list_dir_or_file` returns the path relative to `dir_path`.




**Args:**


 - <b>`dir_path`</b> (str | Path):  Path of the directory.

 - <b>`list_dir`</b> (bool):  List the directories. Default: True.

 - <b>`list_file`</b> (bool):  List the path of files. Default: True.

 - <b>`suffix`</b> (str or tuple[str], optional):   File suffix

 - <b>`that we are interested in. Default`</b>:  None.

 - <b>`recursive`</b> (bool):  If set to True, recursively scan the

 - <b>`directory. Default`</b>:  False.




**Yields:**


 - <b>`Iterable[str]`</b>:  A relative path to `dir_path`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L846"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parse_uri_prefix`

```python
parse_uri_prefix(uri: Union[str, Path]) → Optional[str]
```

Parse the prefix of a uri.




**Args:**


 - <b>`uri`</b> (str | Path):  Uri to be parsed that contains the file prefix.




**Examples:**

```python
FileClient.parse_uri_prefix('s3://path/of/your/file')
#     's3'
```



**Returns:**


 - <b>`str | None`</b>:  Return the prefix of uri if the uri contains '://' else
`None`.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1022"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `put`

```python
put(obj: bytes, filepath: Union[str, Path]) → None
```

Write data to a given `filepath` with 'wb' mode.




**Note:**

> `put` should create a directory if the directory of `filepath`
 does not exist.




**Args:**


 - <b>`obj`</b> (bytes):  Data to be written.

 - <b>`filepath`</b> (str or Path):  Path to write data.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1035"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `put_text`

```python
put_text(obj: str, filepath: Union[str, Path]) → None
```

Write data to a given `filepath` with 'w' mode.




**Note:**

> `put_text` should create a directory if the directory of
 `filepath` does not exist.




**Args:**


 - <b>`obj`</b> (str):  Data to be written.

 - <b>`filepath`</b> (str or Path):  Path to write data.

 - <b>`encoding`</b> (str, optional):  The encoding format used to open the

 - <b>``filepath`. Default`</b>:  'utf-8'.




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L936"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `register_backend`

```python
register_backend(name, backend=None, force=False, prefixes=None)
```

Register a backend to FileClient.


This method can be used as a normal class method or a decorator.


.. code-block:: python


 class NewBackend(BaseStorageBackend):


 def get(self, filepath):
 return filepath


 def get_text(self, filepath):
 return filepath


 FileClient.register_backend('new', NewBackend)


or


.. code-block:: python


 @FileClient.register_backend('new')
 class NewBackend(BaseStorageBackend):


 def get(self, filepath):
 return filepath


 def get_text(self, filepath):
 return filepath




**Args:**


 - <b>`name`</b> (str):  The name of the registered backend.

 - <b>`backend`</b> (class, optional):  The backend class to be registered,

 - <b>`which must be a subclass of `</b>: class:`BaseStorageBackend`.
When this method is used as a decorator, backend is None.
Defaults to None.

 - <b>`force`</b> (bool, optional):  Whether to override the backend if the name
 has already been registered. Defaults to False.

 - <b>`prefixes`</b> (str or list[str] or tuple[str], optional):  The prefixes

 - <b>`of the registered storage backend. Default`</b>:  None.
`New in version 1.3.15.`




---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/file_client.py#L1050"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove`

```python
remove(filepath: Union[str, Path]) → None
```

Remove a file.




**Args:**


 - <b>`filepath`</b> (str, Path):  Path to be removed.





