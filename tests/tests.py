import torch_kernel as m
import torch

# check version
assert m.__version__ == "0.0.1"

#Create a vector of inputs.
inputs = torch.ones({1, 3, 224, 224})
# Execute the model and turn its output into a tensor.
outputs = m.forward(inputs).toTensor()
if outputs is not None:
    print(outputs.shape)
    assert True
