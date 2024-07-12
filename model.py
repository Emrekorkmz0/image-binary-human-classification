import numpy as np

import tensorflow as tf
import tensorflow.keras.applications.mobilenet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import keras
#from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential

import os

import matplotlib.pyplot as plt

import random



IMG_SAVE_PATH = r'/content/drive/MyDrive/ALL_COLAB_FİLES/person_train'

#Str_to_Int is to one-hot encode the labels
Str_to_Int = {
    'human':0,
    'not_human':1
}

NUM_CLASSES = 2

def str_to_Int_mapper(val):
    return Str_to_Int[val]



import PIL
import cv2

dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    for image in os.listdir(path):
        new_path = os.path.join(path, image)
        try:
            imgpath=PIL.Image.open(new_path)
            imgpath=imgpath.convert('RGB')
            img = np.asarray(imgpath)
            img = cv2.resize(img, (240,240))
            img=img/255.
            dataset.append([img, directory])
        except PIL.UnidentifiedImageError:
            print("Skiping image")
data, labels = zip(*dataset)
temp = list(map(str_to_Int_mapper, labels))
labels = keras.utils.to_categorical(temp)



# Select num_images random images from the testing_data array
indices = random.sample(range(len(data)), 9)
images = np.array(data)[indices]
label = np.array(labels)[indices]

# Create a grid of subplots with the specified dimensions
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

# Plot each image with its corresponding label as the title
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    if label[i][0]==1:
        ax.set_title("Human")
    else:
        ax.set_title("not_human")
    ax.axis('off')

# Display the plot
plt.show()


#########################
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

image = tf.cast(tf.expand_dims(data,0), tf.float32)
image = tf.squeeze(image, axis=0)

data_augmentation(image[1])

######################

plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")

# creating model using densenet201

from keras.applications import DenseNet201
#from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import layers
from keras.layers import Dense, Flatten

densenet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(240,240,3)
)

def build_densenet():
    model = Sequential()
    model.add(densenet)
    model.add(data_augmentation)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    #model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    return model


model = build_densenet()
model.summary()

from keras.callbacks import EarlyStopping

es=EarlyStopping(monitor='val_loss',
                 mode='min',
                 verbose=1,
                 patience=20)


history=model.fit(np.array(data),
                  np.array(labels),
                  epochs = 40,
                  shuffle = True,
                  callbacks=[es],
                  validation_split = 0.2)


# save the model for later use
model.save("/content/drive/MyDrive/ALL_COLAB_FILES/human.h5")


##################
import seaborn as sns
from matplotlib import pyplot

def plot_acc(history):
    sns.set()

    fig = pyplot.figure(0, (12, 4))

    ax = pyplot.subplot(1, 2, 1)
    sns.lineplot(x=history.epoch, y=history.history['accuracy'], label='train')
    sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], label='valid')
    pyplot.title('Accuracy')
    pyplot.tight_layout()

    ax = pyplot.subplot(1, 2, 2)
    sns.lineplot(x=history.epoch, y=history.history['loss'], label='train')
    sns.lineplot(x=history.epoch, y=history.history['val_loss'], label='valid')
    pyplot.title('Loss')
    pyplot.tight_layout()

    pyplot.show()


plot_acc(history)




# Select num_images random images from the testing_data array
indices = random.sample(range(len(testing_data)), 9)
images = np.array(testing_data)[indices]
label = np.array(testing_labels)[indices]

# Create a grid of subplots with the specified dimensions
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

# Plot each image with its corresponding label as the title
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    if label[i][0]==1:
        ax.set_title("human")
    else:
        ax.set_title("not_human")
    ax.axis('off')

# Display the plot
plt.show()




import numpy as np
import cv2
import os
import PIL
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# load model 
model = load_model('human.h5')

# test data and test label
dataset_testing = []
y_true = []

IMG_SAVE_PATH_TESTING = '/content/drive/MyDrive/ALL_COLAB_FİLES/person_validation'

for directory in os.listdir(IMG_SAVE_PATH_TESTING):
    path = os.path.join(IMG_SAVE_PATH_TESTING, directory)
    for image in os.listdir(path):
        new_path = os.path.join(path, image)
        imgpath = PIL.Image.open(new_path)
        imgpath = imgpath.convert('RGB')
        img = np.asarray(imgpath)
        img = cv2.resize(img, (240, 240))
        img = img / 255.0
        dataset_testing.append(img)
        y_true.append(directory)

# converting to numpy array
X_test = np.array(dataset_testing)
y_true = np.array(y_true)

# making prediction
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# convert label as  (0 ve 1)
y_true_binary = np.array([0 if label == 'human' else 1 for label in y_true])
y_pred_binary = np.array([0 if pred == 0 else 1 for pred in y_pred])

# creating matrix
cm = confusion_matrix(y_true_binary, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['human', 'not_human'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

