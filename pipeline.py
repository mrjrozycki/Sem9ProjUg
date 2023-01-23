import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
import random

data_cat = ['train', 'valid', 'test'] # data categories

def get_study_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes, 
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset 
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        BASE_DIR = 'Data/%s/' % (phase)
        conditions = os.listdir(BASE_DIR)
        i = 0
        for condition in tqdm(conditions): # for each condition folder
            for study in os.listdir(BASE_DIR + condition): # for each study in that condition folder
                if condition == "normal":
                    label = 0
                else:
                    # if i%3 == 0:
                    #     continue
                    label = 1
                path = BASE_DIR + condition + '/' + study  # path to this study
                study_data[phase].loc[i] = [path, 1, label] # add new row
                i+=1
    return study_data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx, 0]
        image = pil_loader(img_name)
        label = self.df.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample

def get_dataloaders(data, batch_size=1, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.CenterCrop(600),
                # transforms.Resize((300, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.CenterCrop(600),
            # transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in data_cat}
    return dataloaders

if __name__=='main':
    pass