from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import KFold

from convolutional_neural_network import *
from training_tools import *


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(),
    download = True           
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,                         
    transform = ToTensor()           
)

train_data_only, train_data_validation = torch.utils.data.random_split(train_data, [50000, 10000],generator=torch.Generator().manual_seed(27))

loaders = {
    'train' : torch.utils.data.DataLoader(train_data_only, batch_size=100, shuffle=True, num_workers=2),
    'validation': torch.utils.data.DataLoader(train_data_validation, batch_size=100, shuffle=True, num_workers=2),
    'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)
}

## run with different batch size and dropout configuration
active_functions = ['sigmoid', 'tanh', 'ReLU', 'ELU', 'swish']
layer = [1, 16,32,64,128,256,512]
layer = [1,16]
active_functions = ['sigmoid']
history = {
'accuracy_train': [],
'accuracy_val': [],
'loss_train': [],
'loss_val': []
}
for ac_fn in active_functions:  
    print(ac_fn)
    simple_cnn = SIMPLE_CNN(layers_input_size=layer, activation_function=ac_fn, dropout=0.5, batch_norm=True).to(device)  

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_cnn.parameters(), lr = 0.01)
    print('about to train ... ')
    validation_scores, train_scores = train_epochs(simple_cnn,loss_function, optimizer, loaders['train'], loaders['validation'], device, 2 )

    validation_accuracy = np.array([x[0] for x in validation_scores])
    train_accuracy = np.array([x[0] for x in train_scores])

    validation_loss = np.array([x[1] for x in validation_scores])
    train_loss = np.array([x[1] for x in train_scores])

    history['accuracy_train'].append(train_accuracy)
    history['accuracy_val'].append(validation_accuracy)
    history['loss_train'].append(train_loss)
    history['loss_val'].append(validation_loss)
    accuracy_test, loss_test = compute_accuracy_loss(loaders['test'],simple_cnn, loss_function, device)
    print("TEST ACCURACY", ac_fn)
    print(accuracy_test, accuracy_test*100)
print('*********** TRAIN ***************')
print(history['accuracy_train'])
print('--------- LOSS -----------')  
print(history['loss_train']) 
print('*********** VALIDATION ***************')
print(history['accuracy_val'])
print('--------- LOSS -----------')   
print(history['loss_val'])  
index = 0
title = ', '.join(str(l) for l in layer)
for epochs_data in history['accuracy_val']:
    plt.plot(epochs_data, label=active_functions[index])
    index=index+1
plt.ylim([0.6,1])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('Validation with layers: '+ title)
plt.legend()
plt.show()
index = 0
for epochs_data in history['accuracy_train']:
    plt.plot(epochs_data, label=active_functions[index])
    index=index+1
    plt.ylim([0.6,1])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Training with layers: '+ title)
    plt.legend()
plt.show()



#run all layers, all activation functions batch size 512
loaders = {

    'train' : torch.utils.data.DataLoader(train_data_only, batch_size=512, shuffle=True, num_workers=2),
    'validation': torch.utils.data.DataLoader(train_data_validation, batch_size=512, shuffle=True, num_workers=2),
    'test': torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True, num_workers=2)
}

active_functions = ['sigmoid','tanh','ReLU', 'ELU', 'swish']
layers = [ [1,16,32],[1,16,32,64],[1,16,32,64,128], [1,16,32,64,128 ,256], [1,16,32,64,128 ,256,512]]
epochs = 50


for layer in layers:
  history = {
    'accuracy_train': [],
    'accuracy_val': [],
    'loss_train': [],
    'loss_val': []
  }
  print('processing case: ',layer)
  for ac_fn in active_functions:  
    simple_cnn = SIMPLE_CNN(layers_input_size=layer, activation_function=ac_fn, dropout=0.5, batch_norm=True).to(device)  
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_cnn.parameters(), lr = 0.01)
    
    validation_scores, train_scores = train_epochs(simple_cnn,loss_function, optimizer, loaders['train'], loaders['validation'],device,epochs)
    
    validation_accuracy = np.array([x[0] for x in validation_scores])
    train_accuracy = np.array([x[0] for x in train_scores])

    validation_loss = np.array([x[1] for x in validation_scores])
    train_loss = np.array([x[1] for x in train_scores])

    history['accuracy_train'].append(train_accuracy)
    history['accuracy_val'].append(validation_accuracy)
    history['loss_train'].append(train_loss)
    history['loss_val'].append(validation_loss)
    accuracy_test, loss_test = compute_accuracy_loss(loaders['test'],simple_cnn, loss_function, device)
    print("TEST ACCURACY", ac_fn)
    print(accuracy_test, accuracy_test*100)
  print('*********** TRAIN ***************')
  print(history['accuracy_train'])
  print('--------- LOSS -----------')  
  print(history['loss_train']) 
  print('*********** VALIDATION ***************')
  print(history['accuracy_val'])
  print('--------- LOSS -----------')   
  print(history['loss_val'])  
  index = 0
  title = ', '.join(str(l) for l in layer)
  for epochs_data in history['accuracy_val']:
    plt.plot(epochs_data, label=active_functions[index])
    index=index+1
  plt.ylim([0.6,1])
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.title('Validation with layers: '+ title)
  plt.legend()
  plt.show()
  index = 0
  for epochs_data in history['accuracy_train']:
    plt.plot(epochs_data, label=active_functions[index])
    index=index+1
  plt.ylim([0.6,1])
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.title('Training with layers: '+ title)
  plt.legend()
  plt.show()

##cross validation experiment
active_functions = ['ReLU', 'ELU', 'swish']
layers = [[1,16,32,64,128,256,512] ]
epochs = 50
splits = 3

for layer in layers:
  history = {
    'accuracy_train': [],
    'accuracy_val': [],
    'loss_train': [],
    'loss_val': []
  }
  print('processing case: ',layer)
  for ac_fn in active_functions:
    kfold = KFold(n_splits=splits, shuffle=True, random_state=27)
    average_train_accuracy = np.zeros(epochs,dtype='float')
    average_val_accuracy = np.zeros(epochs,dtype='float')
    average_train_loss = np.zeros(epochs,dtype='float')
    average_val_loss = np.zeros(epochs,dtype='float')
    max_accuracy = 0.0
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):
      print("processing fold ", fold)
      train_sample = torch.utils.data.SubsetRandomSampler(train_ids)
      val_sample = torch.utils.data.SubsetRandomSampler(val_ids)

      train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, sampler=train_sample, num_workers=2)
      val_loader = torch.utils.data.DataLoader(train_data, batch_size=512, sampler=val_sample, num_workers=2)

      simple_cnn = SIMPLE_CNN(layers_input_size=layer, activation_function=ac_fn, dropout=0.5, batch_norm=True).to(device)  

        # loss function
      loss_function = torch.nn.CrossEntropyLoss()
        #  optimization function
      optimizer = torch.optim.Adam(simple_cnn.parameters(), lr = 0.01)
      validation_scores, train_scores = train_epochs(simple_cnn,loss_function, optimizer, train_loader, val_loader , epochs)
      validation_accuracy = np.array([x[0] for x in validation_scores])
      train_accuracy = np.array([x[0] for x in train_scores])

      validation_loss = np.array([x[1] for x in validation_scores])
      train_loss = np.array([x[1] for x in train_scores])

      average_train_accuracy = average_train_accuracy + train_accuracy/float(splits)
      average_val_accuracy = average_val_accuracy + validation_accuracy/float(splits)

      average_train_loss = average_train_loss + train_loss/float(splits)
      average_val_loss = average_val_loss + validation_loss/float(splits)
      actual_model_accuracy = validation_accuracy[-1]
      if(max_accuracy < actual_model_accuracy):
        max_accuracy = actual_model_accuracy
        print('---------better model now:---------- ', fold, ac_fn, max_accuracy )
        path_model = './'+ ac_fn + str(fold)
        torch.save(simple_cnn.state_dict(), path_model)
        accuracy_test, loss_test = compute_accuracy_loss(loaders['test'],simple_cnn, loss_function, device)
        print("TEST ACCURACY", ac_fn)
        print(accuracy_test, accuracy_test*100)
    
    history['accuracy_train'].append(average_train_accuracy)
    history['accuracy_val'].append(average_val_accuracy)
    history['loss_train'].append(average_train_loss)
    history['loss_val'].append(average_val_loss)
  print('*********** TRAIN ***************')
  print(history['accuracy_train'])
  print('--------- LOSS -----------')  
  print(history['loss_train']) 
  print('*********** VALIDATION ***************')
  print(history['accuracy_val'])
  print('--------- LOSS -----------')   
  print(history['loss_val'])  
  index = 0
  title = ', '.join(str(l) for l in layer)
  for epochs_data in history['accuracy_val']:
    plt.plot(epochs_data, label=active_functions[index])
    index=index+1
  plt.ylim([0.6,1])
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.title('Validation with layers: '+ title)
  plt.legend()
  plt.show()
  index = 0
  for epochs_data in history['accuracy_train']:
    plt.plot(epochs_data, label=active_functions[index])
    index=index+1
  plt.ylim([0.6,1])
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.title('Training with layers: '+ title)
  plt.legend()
  plt.show()