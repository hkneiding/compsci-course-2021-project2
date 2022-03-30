import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from trainer_functions import *
from plot_images import *


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]

class AutoEncoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
                nn.Linear(3136, latent_size)
        )
        self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 3136),
                Reshape(-1, 64, 7, 7),
                nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
                Trim(),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoderMaxPool(nn.Module):
    def __init__(self, latent_size):
        super().__init__()


        self.conv1 = nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.maxpool = nn.MaxPool2d((3,3), stride=1, padding=1, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.flatten = nn.Flatten()
        self.linear_enc = nn.Linear(3136, latent_size)

        self.linear_dec = torch.nn.Linear(latent_size, 3136)
        self.reshape = Reshape(-1, 64, 7, 7)
        self.deconv1 = nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1, output_padding=1)
        self.maxunpool = nn.MaxUnpool2d((3,3), stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.trim = Trim()

    def encoder(self, x):

        x = self.conv1(x)
        x, pool_indices_1 = self.maxpool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x, pool_indices_2 = self.maxpool(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear_enc(x)

        return x, pool_indices_1, pool_indices_2

    def decoder(self, x, pool_indices_1, pool_indices_2):

        x = self.linear_dec(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = F.relu(x)
        #x = self.maxunpool(x, pool_indices_2)
        x = self.deconv2(x)
        x = F.relu(x)
        #x = self.maxunpool(x, pool_indices_1)
        x = self.deconv3(x)
        x = self.trim(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x):
        x, pool_indices_1, pool_indices_2 = self.encoder(x)
        x = self.decoder(x, pool_indices_1, pool_indices_2)
        return x


class Net(torch.nn.Module):

    def __init__(self, input_size, autoenc_model, hidden_size=10):
        super(Net, self).__init__()

        self.autoenc_model = autoenc_model

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.autoenc_model.encoder(x)[0]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=0)

        return x

def main():

    # setup dataset
    dataset = torchvision.datasets.FashionMNIST('./', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    latent_space_size = 16

    # set up model
    autoenc_model = AutoEncoderMaxPool(latent_space_size)
    autoenc_model.load_state_dict(torch.load('/home/hkneiding/Desktop/autoenc-res/Project-2/autoenc_maxpool_model_latent_' + str(latent_space_size) + '.model', map_location=torch.device('cpu')))
    autoenc_model.eval()
    predict_model = Net(latent_space_size, autoenc_model, hidden_size=latent_space_size)
    predict_model.load_state_dict(torch.load('/home/hkneiding/Desktop/autoenc-res/Project-2/predict_autoenc_maxpool_model_latent_' + str(latent_space_size) + '.model', map_location=torch.device('cpu')))
    predict_model.eval()

    # make predictions and calculate performance
    correct = 0
    for (data, target) in test_loader:

        data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        output = predict_model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print(100 * correct / len(test_loader.dataset))


if __name__ == '__main__':
    main()
