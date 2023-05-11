#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import torch
import torch.optim as optim
import torchvision as tv
import os


from image_proc import *
from nst_functions import *


#Check whether we use a GPU or a CPU
if torch.cuda.is_available():
  print("We are using GPU")
  device = torch.device("cuda")
else:
  print("We are using CPU")
  device = torch.device("cpu")

vgg = tv.models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

vgg.to(device)


# for name, module in vgg.named_children():
#     print(f"Name: {name}, module: {module}")


content_layers_vgg = ['34']
style_layers_vgg = ['2', '7', '12', '21', '28']

content_weight = 1  
style_weight = 1 

OUTPUT_FOLDER = "./eval_outputs/eval_outputs_vgg/"
CONTENT_INPUT_FOLDER = "./eval_inputs_content/"
STYLE_INPUT_FOLDER = "./eval_inputs_style/"

def vgg_nst(content_path, style_path, epochs, steps_per_epoch, show=False):
   #Loading images and moving them to CPU or GPU
    content_image = image_loader(content_path).to(device)
    style_image = image_loader(style_path).to(device)
    content_features_vgg = get_features(content_image, vgg, content_layers_vgg)
    style_features_vgg = get_features(style_image, vgg, style_layers_vgg)
    style_features_vgg = [gram_matrix(feature) for feature in style_features_vgg]
    target_image = content_image.clone().requires_grad_(True).to(device)
    opt = optim.Adam([target_image], lr=0.003)
    #VGG
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            nst_step(target_image, vgg, content_layers_vgg, 
                    style_layers_vgg, content_features_vgg, style_features_vgg,
                    content_weight, style_weight, opt)
    output_path = OUTPUT_FOLDER + content_path.split("/")[-1] + "--" + style_path.split("/")[-1]
    output = tensor_to_image(target_image)
    tv.utils.save_image(target_image, output_path)
    if show:
        img = Image.open(output_path)
        img.show()


def main():
    
    content_paths = []
    for dirpath, dirnames, filenames in os.walk(CONTENT_INPUT_FOLDER):
        for file in filenames:
            content_paths.append(os.path.join(dirpath, file))
    style_paths = []
    for dirpath, dirnames, filenames in os.walk(STYLE_INPUT_FOLDER):
        for file in filenames:
            style_paths.append(os.path.join(dirpath, file))

    content_paths = sorted(content_paths)
    style_paths = sorted(style_paths)

    assert(len(content_paths) == len(style_paths))


    for i in range(len(content_paths)):
        vgg_nst(content_paths[i], style_paths[i], epochs = 10, steps_per_epoch = 100, show = True)
   

if __name__ == "__main__":
    main()




