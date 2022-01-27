# Handwritten-digits-recognition-using-CNN
A Convolutional Neural Network is built using the Tensorflow-Keras framework. The MNIST dataset is used to build this model. This dataset has 70,000 handwritten samples of numbers from 0-9. Each image in the dataset is of 28x28 grayscale pixels size.

## Dataset
MNIST Data set can be imported as from **tensorflow.keras.datasets import mnist**. This dataset comprises of 70,000 images.The size of each image is 28 * 28 pixels and each pixel has a value in the range of 0 to 255 where, 0 pixel value represents black colour and 255 pixel value represents white colour.

## Overview
This project is completed in TensorFlow environment. In the first step I have installed the required libraries which will be used in completing this project.
After installing libraries I have imported MNIST (Modified National Institute of Standards and Technology database) from Keras library.

## Procedure

### 1. Reshaping the data

We are considering the data in the form of 2D images (28x28 pixels) instead of a flattened stream of 784 pixels. As, we have greyscale images so dimension of colour
will be set to 1. The resultant data would be of the form **(60000 * 28 * 28 * 1)**, where 1 signifies greyscale images and 60000 signifies total number of instances in training set. 


### 2. Normalizing Pixel values
As we have pixel values from 0 to 255, so, I am normalizing feature data (X_train and X_test) by dividing each pixel value with maximum possible pixel value.
Instead of using MinMaxScalar for Normalizing feature data, I have divided all the pixel values by maximum pixel value (i.e., 255), because MinMaxScalar requires a 2-D array to normalize data. However, I have previously reshaped the data into 28 * 28 * 1 form.

### 3. Converting train and test labels in One-Hot Encoding format
I have used **np_utils** package to convert train and test labels in One-Hot Encoding format. After performing this step we will get 10 labels (from 0-9) and each image will be labelled between 0-9.

### 4. Building Convolutional Neural Network
I have used Sequential model to define the model. In sequential model we can add each layer sequentially. Brief description about the model:  
- 2 Convolutional Layer with ReLU activation function.  
- MaxPool2D layer (It is used to reduce the size of image by keeping only important features from the image).  
- 1st Dropout Layer to prevent Overfitting.
- Flattening layer (It is used to transform 2D array to 1-D which will be used as input for Neural Network).  
- 2nd Dropout Layer to prevent Overfitting.  
- Dense Layer or Full Connection Layer.  
- Output Layer has 10 nodes. I have used sofmax activation function because there are total 10 classes and I want sum of probabilities to be 1.  
- Model Compilation using adam optimizer and loss = 'categorical_crossentropy' instead of 'binary_crossentropy' because there are 10 different classes. 
- ![image](https://github.com/ManjinderSingh3/Handwritten-digits-recognition-using-CNN/blob/main/Results/Model%20Training.png) 

### 5. Model Description:
a. Input Image shape : (28 * 28 * 1)  
b. 1st Convolution layer : (3,3) kernel size, Number of filter =32, Stride length=1 c. Output of 1st Convolution layer: (28 – 3) +1 = 26  
d. Outputof2ndConvolutionlayer:(26–3)+1 = 24  
e. Output of MaxPool2D Layer: (24-2)/2 +1 = 12  
f. Flatten Layer: 12*12*64 = 9216  
So, Output of 1st convolution layer is (26*26*32), where 32 is number of filters and output of 2nd convolution layer is (24*24*64), where 64 is number of filters.   Similarly, output of all other layers is also calculated in the same way.

### 6. Model Evaluation 
I have trained the model with batch size of 32. For testing the results of model, I took 10 epochs in first attempt and 12 epochs in second attempt.
**Number of epochs : 12**
a. Loss after 12th Epoch - 0.0083  
b. Accuracy after 12th Epoch - 0.9975 (or 99.75%)  
c. Validation/Test loss after 12th Epoch – 0.0489  
d. Validation/Test accuracy after 12th Epoch – 0.9886 (or 98.86%)  

### 7. Observation 
After 10 epochs validation loss increase and validation accuracy decreases.
![image](https://github.com/ManjinderSingh3/Handwritten-digits-recognition-using-CNN/blob/main/Results/Model%20Accuracy%20and%20loss.png)

### 8. Prediction
a. Randomly chosen digit
![image](https://github.com/ManjinderSingh3/Handwritten-digits-recognition-using-CNN/blob/main/Results/Prediction.png)
#### Confusion Matrix
![image](https://github.com/ManjinderSingh3/Handwritten-digits-recognition-using-CNN/blob/main/Results/Confusion%20Matrix.png)

#### Confusion Matrix Explanation:
Size of Confusion Matrix = (Number of unique classes * Number of unique classes). In my testing dataset there are 10000 rows. Results of my model are:  
![image](https://github.com/ManjinderSingh3/Handwritten-digits-recognition-using-CNN/blob/main/Results/CM%20Explanation.png)
