from keras.preprocessing.image import ImageDataGenerator

#-----------------------------------------
import os
import cv2
import numpy as np

img_width = 224

# load data
label_cnt, X, y = 0, [], []
for folder in os.listdir("data/flower_photos"):

    print('loading: {0}'.format(folder))

    for filename in os.listdir("data/flower_photos/" + str(folder)):
        img = cv2.imread("data/flower_photos/" + str(folder) + '/' + str(filename))

        # resize and append
        X.append(cv2.resize(img, (img_width, img_width)))
        y.append(label_cnt)

    # new label for new folder
    label_cnt += 1

#-----------------------------------------

# show random image from set
import matplotlib.pyplot as plt
rand = np.random.randint(0, len(X))
rgb = cv2.cvtColor(np.array(X[rand]),cv2.COLOR_BGR2RGB)
imgplot = plt.imshow(rgb)
plt.show()

# scale and cast to np array as float32
X = (np.array(X) / 255).astype(np.float32)
y = np.array(y)

#-----------------------------------------
from keras.utils import to_categorical

# shuffle
np.random.seed(42)
shuffle_index = np.random.permutation(len(X))
X, y = X[shuffle_index], y[shuffle_index]

# split train and test data
train_size = 0.8
idx = int(train_size*X.shape[0])
X_train, X_test = X[:idx], X[idx:]
y_train, y_test = y[:idx], y[idx:]

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, label_cnt)
y_test = to_categorical(y_test, label_cnt)

print("train: {0} | test: {1}".format(X_train.shape[0], X_test.shape[0]))

#-----------------------------------------

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D

# reconstructing VGG19 model
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(img_width, img_width, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

# load pretrained weights
model.load_weights('models/vgg19_weights.h5')

# freeze the pretrained layers
for layer in model.layers[:5]:
    layer.trainable = False

# adding custom layers 
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1024, activation="relu"))
model.add(Dense(label_cnt, activation="softmax"))

# compiling the model
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

#-----------------------------------------

# Train the model 
model.fit(x=X_train, y=y_train,
                batch_size=16,
                epochs = 10,
                shuffle=True)

print('Test accuracy: {0:.3}%'.format(model.evaluate(x=X_test, y=y_test, batch_size=16)[1] * 100))