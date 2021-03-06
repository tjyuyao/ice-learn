# Release Versions

自 v0.3.1 以上的版本，要逐渐开始考虑功能的稳定性和向下兼容性，并在 v1.0.0 版本稳定化接口。在 v1.0.0 版本以后，则需要尽可能严格地遵循 [语义化版本号](https://semver.org/) 的规范，即：

```
版本格式：主版本号.次版本号.修订号，版本号递增规则如下：

主版本号：当你做了不兼容的 API 修改，
次版本号：当你做了向下兼容的功能性新增，
修订号：当你做了向下兼容的问题修正。
先行版本号及版本编译信息可以加到“主版本号.次版本号.修订号”的后面，作为延伸。
```

## Release Notes

### v0.4.0 Eager mode, seperate cutex and support for PyTorch 1.11

- 允许在主进程中直接执行结点（不通过 run 接口，而是直接执行 forward），这有利于对结点的单元测试。
- 将对 PyCUDA 的支持移出到 cutex 单独维护， ice-learn 核心处理好顶层接口、多任务和并发相关的问题。
- 新增支持兼容 PyTorch 1.11。
- 包含 numpy array 的列表不再被自动拼接为 minibatch。这是为了保留原始数据以便于CPU计算或可视化等操作，转为 Tensor 的操作需要根据数据类型不同而由用户显式地指定。

### v0.3.1  The first release on pypi.

由于在最基础的代码结构确定下来之前的各种尝试中，一共重写了三次代码，故称之为 0.3 版本。之前两版的代码在不同的仓库里面，之后单独建立一个归档的链接加在此处，以备查阅。（TODO）