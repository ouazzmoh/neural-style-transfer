import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

import net 


def image_loader(path):
    """
        Preprocessing of the images and converting to tensors 
    """
    image = Image.open(path)
    loader = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
    ])
    image = loader(image).unsqueeze(0)
    return image


def AdaIN(content_features, style_features):
    """
        Main function that does the style transfer:AdaIN receives a content
        input x and a style input y, and simply aligns the channelwise 
        mean and variance of x to match those of y.
    """
    content_mean = content_features.mean(dim=(2, 3), keepdim=True)
    content_std = content_features.std(dim=(2, 3), keepdim=True) + 1e-7 #adding small val to avoid zero division
    style_mean = style_features.mean(dim=(2, 3), keepdim=True)
    style_std = style_features.std(dim=(2, 3), keepdim=True) + 1e-7
    normalized_features = (content_features - content_mean) / content_std
    normalized_features = normalized_features*style_std  + style_mean
    return normalized_features




# Load pre-trained models
encoder = net.vgg
decoder = net.decoder

# load params for the nmodels
encoder.load_state_dict(torch.load("auto-encoder-nst/models/vgg_normalised.pth"))
decoder.load_state_dict(torch.load("auto-encoder-nst/models/decoder.pth"))

#We extract 31 layers from the pretrained VGG
encoder = torch.nn.Sequential(*list(encoder.children())[:31])



OUTPUT_FOLDER = "./eval_outputs/eval_outputs_vae/"


def style_transfer_vae(content_path, style_path, show = False, alpha=0.5):

    content_image = image_loader(content_path)
    style_image = image_loader(style_path)

    # Perform style transfer
    content_features = encoder(content_image)
    style_features = encoder(style_image)

    with torch.no_grad():
        normalized = AdaIN(content_features, style_features)
        normalized = normalized * alpha + content_features * (1 - alpha)
        output = decoder(normalized)
    output_path = OUTPUT_FOLDER + content_path.split("/")[-1] + "--" + style_path.split("/")[-1]
    save_image(output, output_path)
    if show:
        img = Image.open(output_path)
        img.show()


CONTENT_INPUT_FOLDER = "./eval_inputs_content/"
STYLE_INPUT_FOLDER = "./eval_inputs_style/"


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
    style_transfer_vae(content_paths[i], style_paths[i], show = True, alpha=0.9)



