## Custom `torch` C++ Kernel
---

- Compiled for Python 3.6.x
- Custom functions written in C++ v14, using `torch` v1.x C++ API
- Functions bound to a Python module using `pybind11`
- Uses `setuptools` for building, and `pytest` for testing

To compile, run
```bash
pip install -r requirements.txt
python setup.py build
```
This will also install the module `torch_kernel` in the Python system path.

To run the unit tests, run
```bash
pytest
```

To import the module, import first `torch`
```python
import torch
import torch_kernel
```
