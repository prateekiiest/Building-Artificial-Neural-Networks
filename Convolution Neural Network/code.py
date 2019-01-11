"""
Flower Recognition using Convolution Neural Network
@author : Prateek Chanda

In this example, we are trying to classify flowers based on specific image features using convolution neural network. 
We compare the results of classification done using both normal images and the grayscale images.
"""

# We use keras model for implementing the convolution neural network model.
# We also use tensorflow for image generation purposes.
import numpy as np
from keras import layers
from keras import models
from PIL import Image 
import seaborn as sns
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

# We first perform the classification procedure for the original set of images

# Split images into Training and Validation (90%)
train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, 
                           zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.1)

img_size = 120
batch_size = 15
t_steps = 3462/batch_size
v_steps = 861/batch_size
#dev_steps = 861/batch_size

train_gen = train.flow_from_directory("../input/flowers/flowers", target_size = (img_size, img_size), 
                                      batch_size = batch_size, class_mode='categorical', subset='training')
valid_gen = train.flow_from_directory("../input/flowers/flowers/", target_size = (img_size, img_size), 
                                      batch_size = batch_size, class_mode = 'categorical', subset='validation')



# We plot the first few images from the dataset
img1 = "../input/flowers/flowers/daisy/100080576_f52e8ee070_n.jpg"
img2 = "../input/flowers/flowers/dandelion/10043234166_e6dd915111_n.jpg"
img3 = "../input/flowers/flowers/sunflower/10386503264_e05387e1f7_m.jpg"
img4 = "../input/flowers/flowers/rose/10503217854_e66a804309.jpg"

imgs = [img1, img2 , img3, img4]
f, ax = plt.subplots(1, 4)
f.set_size_inches(80, 40)
for i in range(4):
    ax[i].imshow(Image.open(imgs[i]).resize((120, 120), Image.ANTIALIAS))
plt.show()

# Image Plot : https://github.com/prateekiiest/Building-Artificial-Neural-Networks/blob/master/Convolution%20Neural%20Network/plots/flower.png

# Here, we develop the CNN Model

# We first add 2 convolution neural network layers using the relu as activation function and we finally add
# a top fully connected layer on top of it for classification.
# In between the layers, we use Max Pooling.
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

# Running the model 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=20, validation_data=valid_gen, validation_steps=v_steps)
model.save('flowers_model.h5')


# Lastly we plot the accuracy and loss for the model
acc = model_hist.history['acc']
val_acc = model_hist.history['val_acc']
loss = model_hist.history['loss']
val_loss = model_hist.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15, 6));
plt.subplot(1,2,1)
plt.plot(epochs, acc, color='#0984e3',marker='o',linestyle='none',label='Training Accuracy')
plt.plot(epochs, val_acc, color='#0984e3',label='Validation Accuracy')
plt.title('Training and Development Accuracy')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss, color='#eb4d4b', marker='o',linestyle='none',label='Training Loss')
plt.plot(epochs, val_loss, color='#eb4d4b',label='Validation Loss')
plt.title('Training and Development Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# Accuracy Plot : https://github.com/prateekiiest/Building-Artificial-Neural-Networks/blob/master/Convolution%20Neural%20Network/plots/trainin.png


# We now perform the classification procedure for the images converted to grayscale

# We first plot the grayscale versions of the images
img1 = "../input/flowers/flowers/daisy/100080576_f52e8ee070_n.jpg"
img2 = "../input/flowers/flowers/dandelion/10043234166_e6dd915111_n.jpg"
img3 = "../input/flowers/flowers/sunflower/10386503264_e05387e1f7_m.jpg"
img4 = "../input/flowers/flowers/rose/10503217854_e66a804309.jpg"

imgs = [img1, img2 , img3, img4]

f, ax = plt.subplots(1, 4)
f.set_size_inches(80, 40)
for i in range(4):
    ax[i].imshow(Image.open(imgs[i]).resize((120, 120), Image.ANTIALIAS).convert('L'))
    
plt.show()   

# Image Plot : # Image Plot : https://github.com/prateekiiest/Building-Artificial-Neural-Networks/blob/master/Convolution%20Neural%20Network/plots/gray_flower.png


# We then create the Model for the grayscale images
# We first add 2 convolution neural network layers using the relu as activation function and we finally add
# a top fully connected layer on top of it for classification.
# In between the layers, we use Max Pooling.

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

gray_model_hist = model.fit_generator(gray_train_gen, steps_per_epoch=t_steps, epochs=10, 
                                      validation_data=gray_valid_gen, validation_steps=v_steps)
model.save('gray_flowers_model.h5')


# Lastly we plot the accuracy and loss for the model
acc = gray_model_hist.history['acc']
val_acc = gray_model_hist.history['val_acc']
loss = gray_model_hist.history['loss']
val_loss = gray_model_hist.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15, 6));
plt.subplot(1,2,1)
plt.plot(epochs, acc, color='#0984e3',marker='o', linestyle='none',label='Training Accuracy')
plt.plot(epochs, val_acc, color='#0984e3',label='Validation Accuracy')
plt.title('Training and Development Accuracy')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none',label='Training Loss')
plt.plot(epochs, val_loss, color='#eb4d4b',label='Validation Loss')
plt.title('Training and Development Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# Accuracy Plot : # Image Plot : https://github.com/prateekiiest/Building-Artificial-Neural-Networks/blob/master/Convolution%20Neural%20Network/plots/gray_plot.png
