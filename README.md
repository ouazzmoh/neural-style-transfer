# Neural Style Transfer

## Overview

This project implements **Neural Style Transfer**, a technique used to apply the artistic style of one image to the content of another. By leveraging Convolutional Neural Networks (CNNs), specifically the **VGG19** model, the project generates images that combine content from one image with the style from another. The result is an image that looks like it has been painted in the style of a famous artwork.

The project also explores other approaches to style transfer, such as **AdaIN** (Adaptive Instance Normalization) for faster style transfer, and investigates the potential of using **CycleGANs** for unpaired image translation.

## Key Components

### 1. **VGG19 Classifier**
- **Purpose**: Serves as the backbone for extracting content and style features.
- **Methodology**: The VGG19 model is used to extract intermediate representations from images, which are then used to calculate content and style losses.
  
### 2. **Home-Trained Classifier**
- **Custom CNN**: A convolutional neural network built and trained from scratch using a smaller dataset. While not as effective as VGG19, it was designed to explore trade-offs between accuracy and training time.
  
### 3. **AdaIN (Adaptive Instance Normalization)**
- **Fast Neural Style Transfer**: AdaIN adjusts the statistical properties of the content image to match those of the style image for faster style transfer. It uses an encoder-decoder architecture for real-time image generation.

### 4. **CycleGAN**
- **Unpaired Image-to-Image Translation**: Although not fully explored due to computational constraints, CycleGAN offers a method for performing style transfer without paired images. This technique is promising for future extensions.

## How It Works

1. **Content and Style Extraction**: 
   - VGG19 is used to extract features from the content and style images at different convolutional layers. 
   - The content image’s deep features are compared with the generated image to calculate the **content loss**.
   - The style image’s features are used to compute the **Gram matrix**, which is compared with the generated image’s features to compute the **style loss**.

2. **Optimization**: 
   - The neural network aims to minimize a total loss function that is a weighted sum of the content loss and style loss, using gradient descent to iteratively update the generated image.

3. **AdaIN**:
   - This method speeds up the style transfer by normalizing the content image’s feature maps to match the style image’s statistics, thus avoiding complex iterative processes.
