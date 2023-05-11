Computer Vision and Pattern Recognition
=========================================

Practical 1: Introduction to Neural Networks
=========================================

### Goals
- Get familiar with PyTorch
- Understand a training and evaluation pipeline
- Create a deep convolution encoder
- Use a pre-trained network
- Create a bounding box regressor
- Compare different architectures

### Evaluation
The practical will be evaluated in groups of two.
It is expected that you write a report with the plots / results of the 
different experiments. The code must be submitted with the report.

A good report :
* briefly details the experiment setting 
* illustrates the results (with screenshots, tables)
* briefly discusses the outcome of the experiment

```

Reports are due on January 30th 2023


```

### Getting started

The practical will takes place on the Ensimag / Grenoble INP new educational
GPU cluster. It is composed of 75 virtual machines each with access to a NVidia
Turing GPU with 8Gb dedicated VRAM. To access a node of this cluster for 
the purpose of this practical, you must log onto a node from a Unix terminal
with ssh as follows:

```
$ ssh -Y login@vmgpuXXX.ensimag.fr
```

where XXX is a number from 000 to 075, and login is your Ensimag/Agalan login.
A node number will be assigned to each student at the beginning of each
practical. The `-K` option is for Kerberos authentication. 
The `-Y` option ensures that the graphical display is forwarded,
such that you can open graphical windows on the remote node. You can check 
that the display forwarding works with the following command:

```
vmgpu001:$ xclock
```

## Part 1: Discover PyTorch  with a simple first classification network

1. Discover the project layout, dataset and code. 
   - **dataset** : Our toy dataset contains images for 3 categories: 
     `motorcycles`, `airplanes` and `faces`. The dataset in on
     Ensimag machines in folder `/matieres/5MMVORF/01-dataset`, referred
     throughout as `dataset` folder below for simplicity. 
     
   - [`train.py`](train.py) is the file containing the main training code. It first
     retrieves the image paths in `dataset/images` and annotations in 
     `dataset/annotations/`, which contain a folder per label type.
     It then constructs a `ImageDataset` object which abstracts the dataset
     for Pytorch. This `Dataset` object is then accessed through a Pytorch 
     `DataLoader` object which allows to retrieve a data group, or batch
     for the optimization. Typically we will use a batch size of 32 in this
     exercise. We will split the main image dataset into three Datasets: the
     training, validation and test sets which were described in the lectures.
     Then comes the training loop where the magic happens and network
     parameters are optimized.
   - the `model` folder contains the model description:
       - [`network.py`](model/network.py) contains the Pytorch Neural Network descriptions
       - [`config.py`](model/config.py) defines the configuration constants and paths
       - [`dataset.py`](model/dataset.py) is where we define our custom PyTorch `Dataset`,
         which basically tells PyTorch where it can retrieve the data an how
         to format it for training.
   - [`predict.py`](predict.py) is a python script to visualize the data and annotations
     in a CSV file. The CSV file can either be one of the dataset files or one
     of the train, validation or test sets generated in the `output/` folder at
     training. It basically loops over all images or CSV files given as
     arguments, runs them through the network, then retrieves and displays
     the predictions.
   - [`eval.py`](eval.py) is a more elaborate evaluation script to run after training.
     It evaluates the performance of a trained model provided as argument
     (stored in `output/{model}.pth`). 
     
     It provides statistics about the overall performance of the training,
     and also per-label statistics which may help you identify imbalances or
     disparities between the labels.
    
   
2. Run the training code (SimpleDetector) `python train.py` in the main
   directory. It loops for 20 epochs, then stores the trained model and 
   parameters after the last epoch in the Pytroch dump file 
   `output/last_model.pth`.

3. Run `python eval.py output/last_model.pth` and observe the numerical 
   results.

4. Run `predict.py output/{dataset_name}.csv` and observe the predictions.
   Add a flag to the application, such that when this flag is provided 
   on the command line, the script only shows the failure cases 
   in the display loop instead of all images.

5. Write some code in [`train.py`](train.py) to create training loss and accuracy plots
   with matplotlib.

6. Write some code in [`train.py`](train.py) to store the model with best validation
   accuracy. In case two models have the same accuracy, select the one with lowest loss.
   
7. Retrain the network and eval again. What can you say about the plot, 
   the selected epoch and the obtained results?
   
8. Try Increasing the number of epochs (up to 200)

9. For each of the next items, adapt the network or configuation, train it, evaluate it by running [`eval.py`](eval.py)
and comment on the obtained results and failure cases:

    9.1 Train on a small train set (1% - 5% - 10%) and small validation set (20%).

    9.2 Try removing the dropout layers and the batch normalizations.

    9.3 Train the configuration with no dropout, no batch normalization, 
    200 epochs,  0.005 for cut_val and 0.10 for cut_test. 
    Compare the results of the model with best validation accuracy and the model form the last epoch.
    What can you see?

10. Download internet images and run them through the network. Show a few
    success and failure cases and comment.


## Part 2: Creating a deeper model + using a pretrained architecture

1. Create another detector class `DeeperDetector` in `model/network.py`. 
   Add layers to it to make it deeper.
   Be careful: MaxPool divides the size of the image by a stride factor.
   In general the kernel size is chosen with the same size as the stride. 
   The Linear layers (fully connected) need to get as input a reasonable 
   number of features: typically 7x7 size of the feature map, multiplied by the
   number of features per pixel, for example 512 to avoid too many combinations. 
   For the output feature map after convolutions and maxpool, the feature map
   needs to be flattened (Flatten).  Train, evaluate and comment.

   **Note**: *the number of input features of a layer needs to match the
   number of output features of the previous layer
   (including after a Flatten operation)*.
   
2. Draw inspiration from the VGG11  architecture (look up on internet or in
   course slides) and further deepen your network. Use 512 features in 
   the deeper layer (instead of the 4096 in the vanilla VGG11).
   Train, evaluate and comment.
 
3. Using pretrained models with Pytorch. It is possible to load pre-trained,
   standard architectures with Pytorch. Look at the given class
   `ResnetObjectDetector` which uses ResNet18. 
   
   The feature extraction layers are simply copied, and we just replace
   the Fully Connected (Linear) layers with our own blocs. Train, evaluate and comment.
   
4. Compare the 4 different models (SimpleModel, DeepModel, VGG-like, Resnet)
   by making a table of loss and accuracy by model.
   Comment on the computation time of an epoch, and the evolution of the loss
   function.
   
   Provide aggregated and refined results per label in the comparison.
   Download internet images and run them through the network. Show a few 
   success and failure cases and comment.
   

## Part 3: Bounding Box Regression

So far we have dealt with a classification problem. Now we are going to
estimate the bounding box of the object. This estimation problem is cast as
a *neural network regression problem*, for which the trained neural network will
provide predictions. The bounding box annotations used for training are given
in the same CSV files as we have used up to now. They include the top left 
and bottom right coordinates of the box. 

For best mutual performance on the classification and regression tasks,
we will make a co-learning network. In such a network, the feature layers will be shared between the two tasks of 
classification and bounding box prediction. For this, we are going to add a 
new second branch to our networks, dedicated to the bounding box regression. 
It will be composed by (Linear, Relu, Linear, Relu, Linear, **Sigmoid**, 
*and no Dropout*). From the same extracted features, each branch
(classification and regression) are going to have their own Fully Connected
layers to perform their specialized task.

You shall start with SimpleNetwork, and once you have better understanding,
propagate similar additions to your other networks.

1. Create the fully connected network for the bounding box regression task in
   your pre-existing networks.  

2. The forward method now needs to be modified, instead of only outputting the
   label, it needs to output the regressed bounding box (return a 4-tuple).
   The 4-tuple needs to be retrieved in [`train.py`](train.py), [`eval.py`](eval.py)
   and [`predict.py`](predict.py):
   `predict = object_detector(images)` will now have two components 
   `predict[0]` and `predict[1]`.
   
3. Add code to read the annotations in [`train.py`](train.py) and in
   [`dataset.py`](model/dataset.py). Be careful, the bounding box coordinates need to be
   normalized to be in [0,1] x [0,1], a ratio with respect the size
   of the image, even though the images are themselves normalized at 224x224 
   (ImageNet Size) before they are passed to the network. 

4. We need to compute the new training loss component for the bounding boxes: 
   modify in [`train.py`](train.py) the `compute_loss` function to get the
   annotations of the bounding box and compute the loss of the prediction. 

5. You can now execute `python train.py` and display bounding box losses to see if 
   things work. 
   
6. Now let's see what happens visually! Modify [`predict.py`](predict.py) to display
   the bounding boxes in images for predicted bounding boxes and the ground
   truth bounding boxes (using `cv2.rectangle`). 
   
7. Resnet can be loaded with pretrained weights (constructor argument 
   `resnet18(pretrained = True`, in [`network.py`](model/network.py)).
   They can be frozen or included in the training to be refined towards the task 
   (transfer learning). For the latter, simply comment the code calling the 
   `param.requires_grad = False` in [`network.py`](model/network.py).
   Test the possibilities and compare.

8. (optional) Do the same for your other Neural Network architectures. 
   Test and compare your different architectures `SimpleDetector`, 
   `DeeperDetector`
   and `ResnetDetector` and plot the loss evolutions on a common error plot
   now adjusted for bounding box prediction. 
   Note your observations on training losses, results, failure cases in a 
   error analysis table.

9. (optional) Download internet images and run them through the network. 
   Show a few success and failure cases and comment.


## Part 4: Comparison with classic approaches (optional)

- Document yourself with OpenCv classic object and face detection approaches.
Execute them on the provided dataset. Compare with ground truth and compute the same loss
that you used for your Networks to have comparable error analysis.
Compare the different approaches with your learning approaches
by putting extra table entries in your error analysis table.

- Check out [Tina Face](https://paperswithcode.com/paper/tinaface-strong-but-simple-baseline-for-face)
  and [Yolo](https://paperswithcode.com/paper/you-only-look-once-unified-real-time-object) architectures and note the specificities.


## Appendix: running this practical on a personal machine

It is optionally possible to execute the TP at home 
(NVidia / CUDA GPU strongly recommended) with Python >= 3.6 and the following
pip packages installed: 
```
pip install opencv-contrib-python torch torchvision matplotlib numpy
```

**DO NOT** execute this command on Ensimag nodes because the native Ubuntu 
installation is already configured with these packages, 
and pytorch is **more than 1Gb in size**.
For personal machine use you will also need to copy the dataset on your
personal machine to be able to tun the practical 
(`/matieres/5MMVORF/01-dataset`).
