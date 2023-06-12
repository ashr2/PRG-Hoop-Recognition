import torch
import torchvision
import torchvision.transforms as transforms
import hoop_dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Normalize dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

set = hoop_dataset.HoopDataset("/Users/ashwathrajesh/UMD-PRG-Hoop-Recognition/tests/hoops", 
                               "/Users/ashwathrajesh/UMD-PRG-Hoop-Recognition/assets/unlabeled2017", 60)

trainset = torch.utils.data.Subset(set, range(0, int(len(set) * 0.7)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torch.utils.data.Subset(set, range(int(len(set) * 0.7), len(set)))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)



class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Original 
        self.encoder_conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=16, 
                      kernel_size=(16, 16), stride=2)
        self.encoder_conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, 
                      kernel_size=(8, 8), stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=1, return_indices=True)
        # self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=1, padding=1)
        self.decoder_conv2d_1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                               kernel_size=(8, 8), stride=2)
        self.decoder_conv2d_2 = nn.ConvTranspose2d(in_channels=16, out_channels=out_channels,
                               kernel_size=(16, 16), stride=2)

    def forward(self, x):
        # Original
        encoded_conv2d_1 = self.encoder_conv2d_1(x)
        encoded_conv2d_2 = self.encoder_conv2d_2(F.relu(encoded_conv2d_1))
        # encoded, indices = self.pool(F.relu(encoded_conv2d_2))
        # decoded_unpool = self.unpool(F.relu(encoded), indices, output_size=encoded_conv2d_2.size())
        decoded_conv2d_1 = self.decoder_conv2d_1(F.relu(encoded_conv2d_2), output_size=encoded_conv2d_1.size())
        decoded = self.decoder_conv2d_2(F.relu(decoded_conv2d_1), output_size=x.size())
        # Use sigmoid to keep output between 0 and 1
        decoded = torch.sigmoid(decoded)
        return decoded
    
# encoder = AutoEncoder(in_channels = 3, out_channels = 1)

# #Define loss function and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = encoder(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')

# PATH = 'Image Transforms/assets'
# torch.save(encoder.state_dict(), PATH)

# dataiter = iter(testloader)
# images, labels = next(dataiter)
