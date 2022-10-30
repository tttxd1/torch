import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

image_size = [1,28,28]

class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(torch.prod(image_size, dtype=torch.int32),64),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(inplace = True),
            nn.Linear(64, 128),
            torch.nn.GELU(inplace=True),
            nn.Linear(128, 256),
            torch.nn.GELU(inplace=True),
            nn.Linear(256, 512),
            torch.nn.GELU(inplace=True),
            nn.Linear(512, 1024),
            torch.nn.GELU(inplace=True),
            nn.Linear(1024, torch.prod(image_size, dtype=torch.int32)),
            nn.Tanh(),
        )

    def forward(self,z):

        output = self.model(z) # z: [bs, 1*28*28]
        image = output.reshape(z.shape[0],*image_size)

        return image

class Discriminator(nn.Module):

    def __init__(self, in_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim,1024),
            torch.nn.GELU(inplace = True),
            nn.Linear(1024, 512),
            torch.nn.GELU(inplace=True),
            nn.Linear(512, 256),
            torch.nn.GELU(inplace=True),
            nn.Linear(256, 128),
            torch.nn.GELU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self,image):

        pro = self.model(image.reshape(image.shape[0], -1))

        return pro

# traning
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                                         ]
                                     ))
batch_size = 32
dataloader =  DataLoader(dataset, batch_size= batch_size, shuffle=True)

generator = Generator()
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()

num_epoch = 100
latent_dim = 64
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_image, _ = mini_batch
        z = torch.randn(batch_size, latent_dim)
        pred_image = generator(z)

        g_optimizer.zero_grad()
        g_loss = loss_fn(discriminator(pred_image), torch.ones(batch_size, 1))
        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        d_loss = 0.5*(loss_fn(discriminator(gt_image), torch.ones(batch_size, 1)) + loss_fn(discriminator(pred_image.detach()), torch.zeros(batch_size, 1)))


# print(dataset[1])
# print(dataset[1][0])