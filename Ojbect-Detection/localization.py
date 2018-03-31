from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2

# load network
model = load_model('models/vgg_retrained.h5')

# define window size and shift
network_width = 224
window_width = network_width
shift_size = network_width // 4

img = cv2.imread('data/test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("Image shape: {0}".format(img.shape))

x, y = 0, 0 
in_frame = True
box = {'car': [], 'suv': [], 'truck': []}
while in_frame:
    # grab a subsection of image
    sub_img = img[y:y + window_width, x:x + window_width]

    # resize and flip colour space
    sub_img = cv2.resize(sub_img, (network_width, network_width))
    sub_img = np.expand_dims(sub_img, axis=0)

    # classify image and check for match
    pred = model.predict((sub_img / 255).astype(np.float16))
    if pred[0][0] > 0.95:
        box['car'].append([x, y, window_width])
    if pred[0][1] > 0.95:
        box['suv'].append([x, y, window_width])
    if pred[0][2] > 0.95:
        box['truck'].append([x, y, window_width])

    # shift the looking window
    x += shift_size

    # make sure x is within frame
    if x + window_width > img.shape[1]:
        x = img.shape[1] - window_width

    # if at the end of the x axis
    if x == img.shape[1] - window_width:
        
        # window has gone over entire image
        if y == img.shape[0] - window_width:
            in_frame = False

        # reset horizontal axis and slide down
        x = 0
        y += shift_size
    
    # make sure y is within frame
    if y + window_width > img.shape[0]:
        y = img.shape[0] - window_width

# draw bounding box
for item in box['car']:
    cv2.rectangle(img, (item[0], item[1]), (item[0] + item[2], item[1] +  item[2]), (0, 255, 0), 3)
for item in box['suv']:
    cv2.rectangle(img, (item[0], item[1]), (item[0] + item[2], item[1] +  item[2]), (255, 0, 0), 3)
for item in box['truck']:
    cv2.rectangle(img, (item[0], item[1]), (item[0] + item[2], item[1] +  item[2]), (0, 0, 255), 3)

imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
plt.show()
