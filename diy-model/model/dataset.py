from model import config
import torch
from torch.utils.data import Dataset
import cv2
import os

class ImageDataset(Dataset):
    # initialize the constructor
    def __init__(self, data, transforms=None):
        self.transforms = transforms
        self.data = data

    def __getitem__(self, index):
        # retrieve annotations from stored list
        # TODO: retrieve bounding box labels
        filename, x1, y1, x2, y2, label = self.data[index]

        # get full path of filename
        image_path = os.path.join(config.IMAGES_PATH, label, filename)

        # load the image (in OpenCV format), and grab its dimensions
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # scale bounding box coordinates relative to dimensions of input image
        x1 = int(x1) / w
        y1 = int(y1) / h
        x2 = int(x2) / w
        y2 = int(y2) / h
        bbox = torch.tensor([x1, y1, x2, y2])

        # x1 = int(int(x1)/w)
        # y1 = int(int(y1)/h)
        # x2 = int(int(x2)/w)
        # y2 = int(int(y2)/h)

        # normalize label in (0, 1, 2) and convert to tensor
        label = torch.tensor(config.LABELS.index(label))

        # apply image transformations if any
        if self.transforms:
            image = self.transforms(image)

        # return a tuple of the images, labels, and bounding box coordinates
        return image, label, bbox

    def __len__(self):
        # return the size of the dataset
        return len(self.data)