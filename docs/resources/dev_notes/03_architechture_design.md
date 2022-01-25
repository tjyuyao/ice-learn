# The Folder Structure

- `llutil` for Low-Level Utilities that supports the most fundamental mechanisms.
    - `multiprocessing` enables cross-processes communication for tensors, lambdas and closures.
    - `config` converts every class into a `configurable`.
    - `modifier` defines a tool for pipeline operation, useful in dataset transform and learning rate update, etc.

- `core` implements the kernel hyper-graph architecture for scheduling multitasks and experiments.

- `api` organizes all resources and provide user-friendly interfaces.

- `repro` reproduces public works using low-level or high-level APIs in `ice`. Models, tricks, and other useful code will be incorporated back into `api` if in need.
