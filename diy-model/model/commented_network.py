from torch import nn
from torchvision.models import resnet18


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class SimpleDetector(nn.Module):
    """ VGG11 inspired feature extraction layers """
    def __init__(self, nb_classes):
        """ initialize the network """
        super().__init__()
        # TODO: play with simplifications of this network
        self.features = nn.Sequential(
            # 3 input channels = RGB, 32 feature maps, kernel_size (matrix 3x3) li kadoz 3la
            # input wa7d b wa7d, padding: katzid 0 wat li katzid f borders 9bl madowz convolution
            # filter
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),

            """
            The nn.BatchNorm2d layers are batch normalization layers that normalize the outputs 
            of the convolutional layers to improve the stability and speed of
            training the neural network. ALSO IT REDUCES OVERFITTING
            NORMALIZING MAKES TRAINING MORE EFFICIENT
            new value is (old value - mean) / standard deviation
            y = (x - mean) / sqrt(var + eps) * gamma + beta
            THIS IS DONE FOR EACH CHANNEL (FEATURE MAP) INDEPENDENTLY
            """
            nn.BatchNorm2d(32),

            """
            Rectified Linear Unit (ReLU) activation function
            1. Non Linearity: output of CNN is linear combination of input values and kernel wieghts.
                            app:y ReLU(x) = max(0, x) -> NON LINEARITY
            -> enables network to learn more complex and non linear relationships between input and output
            """
            nn.ReLU(),


            """
            downsample the feature maps by a factor of 4
            pooling layer: max pooling
            1. pooling window
            2. non overlapping pooling regions
            3. translation invariance:
                For example, in object recognition tasks, it is important for a model to be 
                invariant to changes in the position, orientation, and scale of the object 
                within the image. If the model were not invariant to these transformations, it 
                would have to learn to recognize the object in each possible position, orientation,
                and scale separately, which would require a very large number of parameters and make
                the model difficult to train.
            4. dimension reduction: helps to reduce the number of parameters in the model
                                 and prevent overfitting, 

            Stride is a hyperparameter that determines how much convolutional kernel is shifted
            between each step
            -> stride size affects size of output. larger stride -> smaller output.
            -> stride size affects number of parameters in model. larger stride -> fewer parameters.
            -> smaller stride -> larger output -> more parameters
            
            There is a trade-off between stride and kernel size.
            """
            nn.MaxPool2d(kernel_size=4, stride=4),


            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            """
            used to flatten the output of the max-pooling layers into a 1D vector, which can then be fed 
            into the classifier layers of the neural network.
            """
            nn.Flatten()
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        # TODO: play with dimensions of this network and compare
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/4^3
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, nb_classes)
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        # TODO: take inspiration from above without dropouts

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        # TODO: compute and add the bounding box regressor term
        return self.classifier(x)

# TODO: create a new class based on SimpleDetector to create a deeper model
class DeeperDetector(SimpleDetector):
    def __init__(self, nb_classes):
        super().__init__(nb_classes)

# TODO: once played with VGG, play with this
class ResnetObjectDetector(nn.Module):
    """ Resnet18 based feature extraction layers """
    def __init__(self, nb_classes):
        super().__init__()
        # copy resnet up to the last conv layer prior to fc layers, and flatten
        # TODO: add pretrained=True to get pretrained coefficients: what effect?
        features = list(resnet18(pretrained=True).children())[:9]
        self.features = nn.Sequential(*features, nn.Flatten())

        # TODO: first freeze these layers, then comment this loop to
        #  include them in the training
        # freeze all ResNet18 layers during the training process
        for param in self.features.parameters():
            param.requires_grad = False

        # create classifier path for class label prediction
        # TODO: play with dimensions below and see how it compares
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, nb_classes)
        )

        # create regressor path for bounding box coordinates prediction
        # TODO: take inspiration from above without dropouts

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        x = self.features(x)
        # TODO: compute and add the bounding box regressor term
        return self.classifier(x)
    

"""
Convolution uses kernel (weights) (small matrix) to output FEATURE MAP
ReLU: to denoise and “activate” most important features by setting to 0 negative values
Pooling layers (eg max pooling): downsample feature maps and reduce spacial dimensions of
         the data→ retain most important features → reduce overfitting & improve robustness
"""