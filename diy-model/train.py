from model.dataset import ImageDataset
from model.network import SimpleDetector as ObjectDetector
from model import config
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as fun
from torch.optim import Adam
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
import os

from PyQt5.QtCore import QLibraryInfo

if __name__ == '__main__':
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    # initialize the list of data (images), class labels, target bounding
    # box coordinates, and image paths
    print("**** loading dataset...")
    data = []

    # loop over all CSV files in the annotations directory
    for csv_file in os.listdir(config.ANNOTS_PATH):
        csv_file = os.path.join(config.ANNOTS_PATH, csv_file)
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(csv_file).read().strip().split("\n"):
            data.append(row.split(','))

    # randomly partition the data: 80% training, 10% validation, 10% testing
    random.seed(0)
    random.shuffle(data)

    cut_val = int(0.8 * len(data))   # 0.8
    cut_test = int(0.9 * len(data))  # 0.9
    train_data = data[:cut_val]
    val_data = data[cut_val:cut_test]
    test_data = data[cut_test:]

    # create Torch datasets for our training, validation and test data
    train_dataset = ImageDataset(train_data, transforms=config.TRANSFORMS)
    val_dataset = ImageDataset(val_data, transforms=config.TRANSFORMS)
    test_dataset = ImageDataset(test_data, transforms=config.TRANSFORMS)
    print(f"**** {len(train_data)} training, {len(val_data)} validation and "
          f"{len(test_data)} test samples")

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             num_workers=config.NB_WORKERS,
                             pin_memory=config.PIN_MEMORY)

    # save testing image paths to use for evaluating/testing our object detector
    print("**** saving training, validation and testing split data as CSV...")
    with open(config.TEST_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in test_data]))
    with open(config.VAL_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in val_data]))
    with open(config.TRAIN_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in train_data]))

    # create our custom object detector model and upload to the current device
    print("**** initializing network...")
    object_detector = ObjectDetector(len(config.LABELS)).to(config.DEVICE)

    # initialize the optimizer, compile the model, and show the model summary
    optimizer = Adam(object_detector.parameters(), lr=config.INIT_LR)
    print(object_detector)

    # initialize history variables for future plot
    plots = defaultdict(list)

    # function to compute loss over a batch
    def compute_loss(loader, back_prop=False):
        # initialize the total loss and number of correct predictions
        total_loss, correct = 0, 0

        # loop over batches of the training set
        for batch in loader:
            # send the inputs and training annotations to the device

            # batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            # images, boxes, labels = batch['image'], batch['boxes'], batch['labels']

            images, labels, bboxes = [datum.to(config.DEVICE) for datum in batch]


            # perform a forward pass and calculate the training loss
            predict, bbox_predictions = object_detector(images)

            # compute the smooth L1 loss between the predicted and ground truth bounding boxes
            bbox_loss = fun.smooth_l1_loss(bbox_predictions, bboxes, reduction="sum")

            class_loss = fun.cross_entropy(predict, labels, reduction="sum")
            batch_loss = config.BBOXW * bbox_loss + config.LABELW * class_loss

            # zero out the gradients, perform backprop & update the weights
            if back_prop:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            total_loss += batch_loss
            correct_labels = predict.argmax(1) == labels
            correct += correct_labels.type(torch.float).sum().item()

        # return sample-level averages of the loss and accuracy
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    # loop over epochs
    print("**** training the network...")
    prev_val_acc = None
    prev_val_loss = None
    start_time = time.time()
    best_val_acc = 0
    best_val_loss = float('inf')
    for e in range(config.NUM_EPOCHS):
        # set model in training mode & backpropagate train loss for all batches
        object_detector.train()

        # Do not use the returned loss
        # The loss of each batch is computed with a "different network"
        # as the weights are updated per batch
        _, _ = compute_loss(train_loader, back_prop=True)

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode and compute validation loss
            object_detector.eval()
            train_loss, train_acc = compute_loss(train_loader)
            val_loss, val_acc = compute_loss(val_loader)

        # update our training history
        plots['Training loss'].append(train_loss)
        plots['Training class accuracy'].append(train_acc)

        plots['Validation loss'].append(val_loss)
        plots['Validation class accuracy'].append(val_acc)

        # print the model training and validation information
        print(f"**** EPOCH: {e + 1}/{config.NUM_EPOCHS}")
        print(f"Train loss: {train_loss:.8f}, Train accuracy: {train_acc:.8f}")
        print(f"Val loss: {val_loss:.8f}, Val accuracy: {val_acc:.8f}")

        # store model with highest accuracy, lowest loss
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            print("**** saving BEST object detector model...")
            # When a network has dropout and / or batchnorm layers
            # one needs to explicitly set the eval mode before saving
            object_detector.eval()  #IMPORTANT!!!!!!!
            torch.save(object_detector, config.BEST_MODEL_PATH)
        if e == config.NUM_EPOCHS - 1:
            print("**** saving LAST object detector model...")
            object_detector.eval()
            torch.save(object_detector, config.LAST_MODEL_PATH)   

    print("**** saving LAST object detector model...")
    object_detector.eval()
    torch.save(object_detector, config.LAST_MODEL_PATH)

    end_time = time.time()
    print(f"**** total time to train the model: {end_time - start_time:.2f}s")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # build and save matplotlib plot
    plt.plot(plots['Training loss'], label='Training Loss')
    plt.plot(plots['Validation loss'], label='Validation Loss')
    plt.plot(plots['Training class accuracy'], label='Training Class Accuracy')
    plt.plot(plots['Validation class accuracy'], label='Validation Class Accuracy')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend(loc='lower left')

    # save the training plot
    plt.savefig(config.PLOT_PATH)
