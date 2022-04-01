import torch
import torch.nn as nn

class SIMPLE_CNN(nn.Module):

    def __init__(self,layers_input_size=[1, 16, 32], dropout=0.0, batch_norm=False, activation_function='ReLU', kernel_size=5, last_layer_size=10, image_size=28):
        super().__init__()
        sequential = []
        last_step = []
        activ_func = self._activation_function(activation_function)
        pooling = nn.MaxPool2d(kernel_size=2)
        padding = int(kernel_size/2)
        dropout_layer = nn.Dropout2d(dropout)

        for i in range(1, len(layers_input_size)):
            layer = nn.Conv2d(layers_input_size[i-1], layers_input_size[i], kernel_size, 1, padding)
            sequential.append(layer)
            if(i > 2 and image_size >= 14):
              sequential.append(pooling)    
              image_size= int(image_size/2)
            sequential.append(activ_func)
            if(batch_norm):
              sequential.append(nn.BatchNorm2d(layers_input_size[i]))
        if(dropout > 0):
          sequential.append(dropout_layer)


        last_step.append(nn.Linear(layers_input_size[-1]*image_size*image_size, last_layer_size) )

        self.convolutions = nn.Sequential(*sequential)
        self.last_layer = nn.Sequential(*last_step)

    def _activation_function(self, name):
      return {
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'ReLU': nn.ReLU(),
        'ELU': nn.ELU(),
        'swish': nn.SiLU()
      }[name]

    def forward(self, x):   
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)  
        output = self.last_layer(x)
        return output  