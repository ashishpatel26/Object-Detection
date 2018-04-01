from urllib.request import urlopen
import cv2
import os
import numpy as np

"""
This file downloads images from the data_URLs file
Images are loaded and saved to dataset/ folder

The clean() function looks for bad images but
manual curation may still be required

"""

def main(src, dest, jump=0):
    print('Load images from {0}'.format(src.split('/')[-1]))

    # load file with URLs
    file = open(src, "r", encoding="utf8")
    failed = 0
    image_cnt = 0
    line_cnt = 0

    for url in file:
        line_cnt += 1
        print('\rloading line: ' + str(line_cnt), end='')

        try:
            # open url truncating newline
            image_str = urlopen(url[:-1]).read()

            # write image to folder
            f = open(str(dest) + str(image_cnt) + ".jpg", "wb")
            f.write(image_str)
            image_cnt += 1
        except:
            failed += 1

    print("\nSuccessful: ", image_cnt)
    print(" | Failed: ", failed)

def clean(label):
    print('Cleaning up files in: {}'.format(label))
    for filename in os.listdir(label[:-1]):
        try:
            img = cv2.imread(label + filename)

            if int(np.average(img)) > 250:
                os.remove(label + filename)

            if img.shape[0] < 85:
                os.remove(label + filename)

        except:
            os.remove(label + filename)

if __name__ == '__main__':
    main(src="data_URLs/cars.txt", dest="dataset/car/car_")
    main(src="data_URLs/truck.txt", dest="dataset/truck/truck_")
    main(src="data_URLs/suv.txt", dest="dataset/suv/suv_")

    clean('dataset/car/')
    clean('dataset/truck/')
    clean('dataset/suv/')
