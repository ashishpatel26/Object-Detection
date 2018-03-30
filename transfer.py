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
        X.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (img_width, img_width)))
        y.append(label_cnt)

    # new label for new folder
    label_cnt += 1

# show random image from set
import matplotlib.pyplot as plt
rand = np.random.randint(0, len(X))
imgplot = plt.imshow(X[rand])
plt.show()

# scale and cast to np array as float32
X = (np.array(X) / 255).astype(np.float32)
y = np.array(y)

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

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D

# reconstructing VGG19 model
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_width, img_width, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

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

from keras.preprocessing.image import ImageDataGenerator

# data augmentation for training
datagen = ImageDataGenerator(width_shift_range=0.2, 
                             height_shift_range=0.2,  
                             horizontal_flip=True)

datagen.fit(X_train)

# Train the model 
model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                    epochs = 10,
                    shuffle=True)


print('Test accuracy: {0:.3}%'.format(model.evaluate(x=X_test, y=y_test, batch_size=16)[1] * 100))

# save model
model.save('models/vgg_retrained.h5')
