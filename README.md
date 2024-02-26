## Custom PyTorch C++ Kernel

- Compiled for Python 3.6.x
- Custom functions written in C++ v14, using `torch` v1.x C++ API
- Functions bound to a Python module using `pybind11`
- Uses `setuptools` for building, and `pytest` for testing
- Implements in C++ a custom LSTM-like layer (forward and backward passes), and plugs it in a regression
  model, based on these tutorials:
  * https://pytorch.org/tutorials/advanced/cpp_extension.html (PyTorch C++ kernel)
  * https://machinelearningmastery.com/building-a-regression-model-in-pytorch/ (PyTorch regression model)

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

To train a sample regressor neural network (one LSTM-like layer followed by a linear layer) run:
```bash
python applications/custom_nn.py
```
This script contains PyTorch classes wrapping up the custom layer. It also shows
how to run stochastic gradient descent to update the network weights, and solve
a sample Scikit-learn regression problem.
