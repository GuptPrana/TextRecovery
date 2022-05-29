import os 
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd

class trdg_dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super(trdg_dataset, self).__init__()
        self.split = split
        self.imagepath = "./data"
        self.images = os.listdir("./data")
        self.num_images = len(self.images)
        self.imagepaths = list(map(lambda x: os.path.join(self.imagepath, x), self.images))
        self.labelpath = "labels.csv"
        self.transform = transforms.Compose([transforms.RandomAffine(10), #randomauto...
                            transforms.ToTensor(), 
                            transforms.Normalize((0.5,), (0.5,))])
        self.basictransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        imagepath = self.imagepaths[idx]
        img = Image.open(imagepath)
        # Should be done when generating the dataset
        # width, height = img.size
        # if width != 128 or height != 32:
        #     img = img.resize((128, 32))
        # Grayscale
        if self.split == 'train':
            img = self.transform(img)
        else:
            img = self.basictransform(img)
        csv = pd.read_csv(self.labelpath)
        labeltext = csv.iloc[0, idx]
        data = {'image':img, 'label':labeltext}
        return data

        

