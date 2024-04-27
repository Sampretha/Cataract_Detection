import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Importing necessary packages from TensorFlow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from pyimagesearch.resnet import ResNet
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
    help=r"C:\Users\sampr\Desktop\Cataract-Detection-master\cataract")
ap.add_argument("-m", "--model", type=str, required=True,
    help=r"C:\Users\sampr\Desktop\Cataract-Detection-master\model.h5")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help=r"C:\Users\sampr\Desktop\Cataract-Detection-master\plot.png")
args = vars(ap.parse_args())

# Taking the epochs and batch_size as input
epochs = 150  
batch_size = 10  
learning_rate = 0.001 

# Constructing paths after argument parsing
train_path = os.path.sep.join([args["dataset"], "training"])
test_path = os.path.sep.join([args["dataset"], "testing"])
val_path = os.path.sep.join([args["dataset"], "validation"])

totalTrain = len(list(paths.list_images(train_path)))
totalVal = len(list(paths.list_images(val_path)))
totalTest = len(list(paths.list_images(test_path)))

# Generating new images: initializing the training data augmentation object
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest")

# Initializing the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# Initializing the training generator
trainGen = trainAug.flow_from_directory(
    train_path,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size)

# Initializing the validation generator
valGen = valAug.flow_from_directory(
    val_path,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

# Initializing the testing generator
testGen = valAug.flow_from_directory(
    test_path,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

# Initializing our Keras implementation of ResNet model and compiling it
model = ResNet.build(64, 64, 3, 2, (2, 2, 3),
    (32, 64, 128, 256), reg=0.0005)
opt = SGD(learning_rate=learning_rate, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# Training our Keras model
H = model.fit(
    trainGen,
    steps_per_epoch=totalTrain // batch_size,
    validation_data=valGen,
    validation_steps=totalVal // batch_size,
    epochs=epochs)

# Resetting the testing generator and then using our trained model to make predictions on the data
print("[INFO] Evaluating network...")
testGen.reset()
predIdxs = model.predict(testGen, steps=(totalTest // batch_size) + 1)

# For each image in the testing set, finding the index of the label with the corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Showing a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
    target_names=testGen.class_indices.keys()))

# Saving the network to disk
print("[INFO] Serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# Plotting the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
