import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*53*53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x# create a complete CNN

def GetDatasetSize(path):
    num_of_image = {}
    num_of_image['Cancer'] = 0
    for folder in os.listdir(path):
        # Counting the Number of Files in the Folder
        if folder != 'normal':
            num_of_image['Cancer'] += len(os.listdir(os.path.join(path, folder)));
        else:
            num_of_image[folder] = len(os.listdir(os.path.join(path, folder)));
    return num_of_image;
#model = Net()


if __name__ == '__main__':
    train_path = "./Data/train"
    val_path = "./Data/valid"
    test_path = "./Data/test"
        
    train_set = GetDatasetSize(train_path)
    val_set = GetDatasetSize(val_path)
    test_set = GetDatasetSize(test_path)
    print(train_set,"\n\n",val_set,"\n\n",test_set)

    labels = ['Cancer', 'Normal']
    train_list = list(train_set.values())
    val_list = list(val_set.values())
    test_list = list(test_set.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.20  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, train_list, width, label='Train',color = 'red')
    rects2 = ax.bar(x, val_list, width, label='Val', color = 'blue')
    rects3 = ax.bar(x + width, test_list, width, label='Test', color = 'green')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Images Count')
    ax.set_title('Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=15)
    ax.legend()

    ax.bar_label(rects1)
    ax.bar_label(rects2)
    ax.bar_label(rects3)

    fig.tight_layout()

    plt.show()


    #to do - resize images