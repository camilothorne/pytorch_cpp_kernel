import torch
import torch_kernel as m

# check version
# assert m.__version__ == "0.0.1"


def test_tanh():
    #Create a vector of inputs.
    inputs = torch.ones((1, 3, 66, 2), dtype=torch.float)
    # Execute the model and turn its output into a tensor.
    outputs = m.d_tanh(inputs)
    #if outputs is not None:
    print(outputs)
    #assert True 
    #else:
    assert outputs.shape == inputs.shape
            
            
def test_sigmoid():
    #Create a vector of inputs.
    inputs = torch.ones((1, 3, 66, 2), dtype=torch.float)
    # Execute the model and turn its output into a tensor.
    outputs = m.d_sigmoid(inputs)
    #if outputs is not None:
    print(outputs)
    #assert True 
    #else:
    assert outputs.shape == inputs.shape
