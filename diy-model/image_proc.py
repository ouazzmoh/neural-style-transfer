
from PIL import Image
import torch
import torchvision as tv
import numpy as np

def image_loader(path):
    image = Image.open(path)
    preprocessor = tv.transforms.Compose([
        tv.transforms.Resize(size=(512, 512)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocessor(image)
    #Adding the batchsize dimension : (batch_size, channels, height, width)
    image = torch.unsqueeze(image, dim=0)
    return image


def tensor_to_image(tensor):
    """ Display a tensor as an image. """
    tensor = tensor.to("cpu").clone() #Moving the tensor to cpu
    image = tensor.detach().numpy().squeeze() #freeze params and remove the dimension 0 and 
    image = image.transpose(1,2,0) #transpose (H,W,C)
    #Removing normalization
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

