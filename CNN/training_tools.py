import torch
import torch.nn as nn

def compute_accuracy_loss(loader, cnn, loss_function, device):
    cnn.eval()

    count = 0
    correct = 0.
    loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = cnn(images)
            loss+= loss_function(output, labels).item()
            prediction = output.argmax(dim=-1)
            correct += (prediction == labels).sum().item()
            count+= labels.shape[0]
    
    accuracy = correct/ count
    loss/= len(loader)
    #cnn.train()
    return accuracy, loss


def train_epochs(simple_cnn, loss_function, optimizer, train_loader, val_loader,device, num_epochs = 50):
  validation_scores = []
  train_scores = []
  print('training... ')
  for epoch in range(num_epochs):
      loss_score = 0.0
      for i, (images, labels) in enumerate(train_loader):
          images, labels = images.to(device), labels.to(device)
          batch_x = torch.autograd.Variable(images)
          batch_y = torch.autograd.Variable(labels)
          output = simple_cnn(batch_x)
          loss = loss_function(output, batch_y)
          loss_score += loss.item()
          
          optimizer.zero_grad()

          loss.backward()

          optimizer.step()
      

      validation_scores.append(compute_accuracy_loss(val_loader, simple_cnn, loss_function, device))
      train_scores.append(compute_accuracy_loss(train_loader, simple_cnn, loss_function, device)) 
      print('{}, {:.4f}, {:.4f}'.format(epoch + 1, validation_scores[-1][0], train_scores[-1][0]))
  return validation_scores, train_scores