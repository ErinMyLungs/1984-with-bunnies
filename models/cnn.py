# module for cnn in pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, utils
import time
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import pandas as pd
import os

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2,2) window
        x = self.pool(F.relu(self.conv1(x)))
        # if size is a square, can only specify single number
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, self.num_flat_features(x))
        x = x.view(-1, 16 * 5 * 5 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FaceLandmarksDataset(Dataset):
    """ Face Landmarks Dataset inheriting from abstract base class"""

    def __init__(self, csv_file:str, root_dir:str, transform=None):
        """
        :param csv_file: filepath to csv with annotations
        :param root_dir: directory with all images (data/faces)
        :param transform: optional transform method to be applied to sample
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[item, 0])

        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[item, 1:].to_numpy().astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        #h and w swapped for landmark cuz images x axis = 1 and y axis = 0
        landmarks = landmarks * [new_w/w, new_h/h]

        return {'image':img, 'landmarks':landmarks}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h- new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top: top+new_h,
                left: left+new_w]
        landmarks = landmarks - [left, top]
        return {'image':image, 'landmarks':landmarks}

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image = image.transpose((2, 0, 1)) #torch image is C X H X W where numpy is H X W X C
        return {'image':torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

if __name__ == '__main__':
    training = True
    net = Net()
    # PATH = './models/class_net.pth'
    # print('loading neural net params')
    # net.load_state_dict(torch.load(PATH))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if training:
        start = time.time()
        for epoch in range(2):
            running_loss = 0.0
            for idx, data in enumerate(trainloader, 0):
                # grab input and target, [inputs, labels]
                inputs, labels = data

                # zero the gradients
                optimizer.zero_grad()

                # forward backward optimize step
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print the stats
                running_loss += loss.item()

                if idx % 2000 == 1999:
                    print(f'{epoch + 1} loss: {(running_loss/2000):.3f}')
                    running_loss = 0.0
        print('Training finished')
        print(f'Time taken: {time.time() - start}')
    print('beginning predict')
    correct = 0
    total = 0
    with torch.no_grad(): # as only predicting, no need to track gradients
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on 10000 test images: {(100* (correct/total)):.3f}')

    class_correct = [0.] * 10
    class_total = [0.]* 10

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        if class_total[i] != 0:
            print(f'Accuracy of {classes[i]} is: {100* (class_correct[i]/class_total[i])}')
            print()
        else:
            print(f'class {classes[i]} total is 0')
            print()