# CNN
EE104 LAB 8 - 12/2/2021
README

Image recognition with a Convolutional Neural Network (CNN) using different methods to further increase accuracy of the model. Methods include: adding dropouts, data augmentation, and batch normalization.

# Py Packages

!pip install tensorflow

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt


**Possibly may need to also install:**

!pip install keras

!pip install h5py

!pip install Matplotlib

!pip install numpy

# Instructions

Open the .ipynb file in CoLab. 
https://colab.research.google.com/

To open the notebook, select ‘File’ > ‘Open Notebook’ (or ‘Upload Notebook’ if it is located in your computer’s file explorer) on the menu bar.

For faster execution, select ‘Runtime’ > ‘Change Runtime Type’ and change the Hardware accelerator to GPU.

![alt text](https://github.com/everythingallpurple/CNN/blob/EE104%20Lab8%20github%20pics/1.png?raw=true)
![alt text](https://github.com/everythingallpurple/CNN/blob/EE104%20Lab8%20github%20pics/2.png?raw=true)

*Note: Display will be white since default is light mode. To get dark mode, click the settings icon and change the theme to dark mode.

Run each cell sequentially by moving the cursor over the brackets on the left side of the code. To run all cells at once so that you don’t have to manually run one by one, select ‘Runtime’ > ‘Run all’.

When running the cell that contains the compile and train code, the output will display the loss, accuracy, validation accuracy, and validation loss for each epoch (training iteration). The validation losses and accuracies correlate to how well the model can predict. Running all 50 epochs will take ~30 min.
![alt text](https://github.com/everythingallpurple/CNN/blob/EE104%20Lab8%20github%20pics/3.png?raw=true)

When running the next cell, the trained model is evaluated by plotting the training losses and accuracies against the validation losses and accuracies respectively. If there is a huge gap between the training and validation results, the model has been overfitted. To combat overfitting and increase the validation accuracy, layers of dropout, data augmentation, and batch normalization were included. 

Example of overfitting as shown in the left figure vs. result of our model in the right figure:
![alt text](https://github.com/everythingallpurple/CNN/blob/EE104%20Lab8%20github%20pics/4.png?raw=true)


The last cell shows the final accuracy of the entire model. The accuracy achieved from my run:
![alt text](https://github.com/everythingallpurple/CNN/blob/EE104%20Lab8%20github%20pics/5.png?raw=true)

The accuracy can also be seen from the previous cell’s output (the one outputting the training vs validation plots):
![alt text](https://github.com/everythingallpurple/CNN/blob/EE104%20Lab8%20github%20pics/6.png?raw=true)

# Notes

Cifar10 dataset is pre-split into 50k and 10k for training and testing sets
	
CNN deals w/tensors of shape (batch size, image height, image width, depth)

-> Depth sizes are typically 3 for RGB & 1 for greyscale

-> Cifar10 images are 32x32x3 (height x width x depth)

**Other possible data augmentation include:**

layers.RandomCrop(img_height, img_width)

layers.RandomContrast(factor)	# factor b/n 0.0-1.0


Data augmentation can also be done with ImageDataGenerator which generates batches of tensor image data with real-time data augmentation:

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

Batch normalization applies to the layer before it and is best used after activation layers

Dropouts best used after pooling layers
	
Baseline code for this lab which achieves ~70% accuracy:

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=gtyDF0MKUcM7

**Error:**

"sparse_categorical_crossentropy received from_logits=True, but the output argument was produced by a sigmoid or softmax activation and thus does not represent logits

Set ‘from_logits=False’ in the compile and train model cell

