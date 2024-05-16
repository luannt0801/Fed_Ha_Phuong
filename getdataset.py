import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import os
NUM_WORKERS = os.cpu_count()
# datasetname == "braintumor":
# print("brain tumor dataset dang su dung")
train_brain_trans = transforms.Compose([transforms.RandomCrop(128, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 1, hue = 0.5),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                ])
test_brain_trans = transforms.Compose([transforms.RandomCrop(128, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 1, hue = 0.5),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                ])
trainset = torchvision.datasets.ImageFolder(root='D:\\Phuong ham hamm\\Fed_Ha_Phuong\\data\\tumor\\Training', transform= train_brain_trans)
testset = torchvision.datasets.ImageFolder(root='D:\\Phuong ham hamm\\Fed_Ha_Phuong\\data\\tumor\\Testing', transform= test_brain_trans)
trainset.targets = torch.tensor(trainset.targets)
testset.targets = torch.tensor(testset.targets)

train_dataloader_augmented = DataLoader(trainset, 
                                        batch_size=42, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_augmented = DataLoader(testset, 
                                       batch_size=42, 
                                       shuffle=False, 
                                       num_workers=NUM_WORKERS)

# print("trainset: ", trainset.targets)
# print("class name :", trainset.classes)