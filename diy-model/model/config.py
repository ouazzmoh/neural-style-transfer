# import the necessary packages
import torch
import os
from torchvision import transforms

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "/home/stan/imag/2A_sem2/vorf/01-intro-to-neural-nets/dataset"
IMAGES_PATH = os.path.join(BASE_PATH, "images")
ANNOTS_PATH = os.path.join(BASE_PATH, "annotations")

# define the path to the base output directory
BASE_OUTPUT = "output"

# define paths to output model, plot and testing image paths
BEST_MODEL_PATH = os.path.join(BASE_OUTPUT, "best_model.pth")
LAST_MODEL_PATH = os.path.join(BASE_OUTPUT, "last_model.pth")
PLOT_PATH = os.path.join(BASE_OUTPUT, "convergence_plot.png")
TEST_PATH = os.path.join(BASE_OUTPUT, "test_data.csv")
VAL_PATH = os.path.join(BASE_OUTPUT, "val_data.csv")
TRAIN_PATH = os.path.join(BASE_OUTPUT, "training_data.csv")

# determine the current device and based on that set the pin memory flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NB_WORKERS = os.cpu_count() if DEVICE == "cuda" else 0
print("**** using", DEVICE.upper())

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 32

# specify the loss weights
LABELW = 1.0
BBOXW = 1.0
# with initial weight for loss homogeneity, bounding boxes are inaccurate
# BBOXW = 1.0e-3

# label table as python list, defaults to ['motorcycle', 'airplane', 'face']
LABELS = os.listdir(IMAGES_PATH)

# define normalization transforms (size, mean, stddev from ImageNet)
TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
