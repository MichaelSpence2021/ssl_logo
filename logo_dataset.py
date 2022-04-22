import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class logoDet3K(Dataset):
    
    def __init__(self, dataset_path, transform = None):
        self.image_paths = []
        self.y = []
        self.transform = transform
        
        top_labels = [path for path in os.listdir(dataset_path) if path[0] != '.']
        print(top_labels[:10])
        label_index = 0
        logo_index = 0
        for label in top_labels:
            label_path = dataset_path + '/' + label
            logo_paths = [path for path in os.listdir(label_path) if path[0] != '.']
            for logo in logo_paths:
                logo_path = label_path + '/' + logo
                img_paths = [path for path in os.listdir(logo_path) if path.split('.')[-1] == 'jpg']
                for path in img_paths:
                    img_path = logo_path + '/' + path
                    self.image_paths.append(img_path)
                    label_vector = torch.zeros(9)
                    label_vector[label_index] = 1.0
                    self.y.append(logo_index)
                logo_index += 1
                    
            label_index += 1
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        
        image = Image.open(self.image_paths[index])
        image = image.resize((32,32))
        if self.transform != None:
            image = self.transform(image)
            return image, self.y[index]
        return image, self.y[index]
