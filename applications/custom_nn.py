import math, copy
import torch
import numpy as np
from tqdm import tqdm
import torch_kernel as lltm_cpp
from torch.nn.modules.module import _addindent
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
from matplotlib import pyplot as plt


def torch_summarize(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights.
    """ 
    
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)
        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   
    tmpstr = tmpstr + ')'
    return tmpstr


class LLTMFunction(torch.autograd.Function):
    '''
    Wrappers for the C++ functions
    '''
    
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)
        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    '''
    Custom LSTM layer
    '''
    
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
    
    
class networkLLTM(torch.nn.Module):
    '''
    Custom model using custom layer
    '''
    
    def __init__(self, input_features, state_size):
        super(networkLLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.lltm = LLTM(input_features, state_size)
        self.output_layer = torch.nn.Linear(state_size, 1)

    def forward(self, input):
        h       = torch.randn(batch_size, state_size) # hidden state
        C       = torch.randn(batch_size, state_size) # memory cell
        hidden, context = self.lltm(input, (h,C))
        out = self.output_layer(hidden + context)
        return out, (hidden, context)

    
if __name__ == '__main__':
    
    '''
    We test the code on the California housing Scikit-learn dataset.
    '''
    
    # Read and scale data
    data    = fetch_california_housing()
    scaler  = preprocessing.MinMaxScaler()
    X       = scaler.fit_transform(data.data.astype(np.float32)) 
    y       = data.target.astype(np.float32)
     
    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train).reshape(-1, 1)
    X_test  = torch.tensor(X_test)
    y_test  = torch.tensor(y_test).reshape(-1, 1)
    
    # Hyperparams
    out_features    = y_train.shape[1]
    state_size      = 32
    epochs          = 100    
    input_features  = X_train.shape[1]
    data_size       = X_train.shape[0]
    test_size       = X_test.shape[0]
    batch_size      = 128
    my_lr           = 0.001
    my_momentum     = 0.009
    
    # Init network
    rnn             = networkLLTM(input_features, state_size) # initialize network
    
    print("--------------------")     
    print("Model and parameters:")
    print(torch_summarize(rnn))
    
    # Choose optimizer and loss function (MSE)
    optimizer       = torch.optim.SGD(rnn.parameters(), lr=my_lr, momentum=my_momentum)
    mse_loss        = torch.nn.MSELoss()   
     
    # Hold the best model
    best_mse        = np.inf   # init to infinity
    best_weights    = None
    
    # Monitor training
    loss_history    = []
    val_history     = []

    torch.autograd.set_detect_anomaly(True) # check for anomaly in gradients
    
    print("--------------------") 
    print(f"SGD for {epochs} epochs, with batch size {batch_size}:")
    for i in tqdm(range(epochs)):
        
        rnn.train()
        start = 0
        count = 0
        
        while data_size - start > batch_size:
    
            optimizer.zero_grad()
            out, (new_h, new_C) = rnn(X_train[start:start+batch_size]) # forward pass
            loss = mse_loss(out, y_train[start:start+batch_size])
            loss.backward(retain_graph=True, gradient=loss.grad)
            optimizer.step()
        
            start = start + batch_size
            count = count + 1
    
            if ((i%10 == 0) & (count%20 == 0)):
                
                print(f" - slice (train) {start}--{start+batch_size}")            
                print(" - loss at epoch {} and train batch {} is: {:.10f}".format(
                    i, count, loss.item()))
                    
        rnn.eval()
        estart = 0
        ecount = 0
        
        while test_size - estart > batch_size:
            
            y_pred, _ = rnn(X_test[estart:estart+batch_size])
            mse = mse_loss(y_pred, y_test[estart:estart+batch_size])
            
            if ((i%10 == 0) & (ecount%20 == 0)):
                
                loss_history.append(loss.item())
                print(f" - slice (val) {estart}--{estart+batch_size}")            
                print(" - MSE at epoch {} and test batch {} is: {:.10f}".format(
                    i, ecount, mse.item()))
            
                val_history.append(mse.item())
                if mse.item() < best_mse:
                    best_mse = mse.item()
                    best_weights = copy.deepcopy(rnn.state_dict())
            
            estart = estart + batch_size
            ecount = ecount + 1   


    print("--------------------")    
    print("Network gradients:")
    for name, param in rnn.named_parameters():
        print(" - grads: ", name, torch.isfinite(param.grad).all())          
    
           
    print("--------------------") 
    print("MSE (best): %.2f" % best_mse)
    print("RMSE (best): %.2f" % np.sqrt(best_mse))
    print("--------------------") 
    
    
    # Save plots
    plt.plot(loss_history, label="train")
    plt.plot(val_history, label="val")
    plt.ylabel("MSE loss")
    plt.xlabel("Validation batch")
    plt.title(f"Training loss over {epochs} epochs")
    plt.legend(loc="upper right", title="(best={:.2f})".format(best_mse))
    plt.savefig("training-vs-val-loss.png")
    plt.show()