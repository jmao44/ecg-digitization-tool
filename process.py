import sys
import scipy
import cv2 as cv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract


class ECGdigitizer:
    def __init__(self):
        self.digitizer = None

    # Helper function to help display an oversized image
    def display_image(self, image, name):
        if image.shape[0] > 1000:
            image = cv.resize(image, (0, 0), fx=0.85, fy=0.85)
        cv.imshow(name, image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Helper function to sharpen the image
    def sharpen(self, img):
        kernel = np.array([[0, -1, 0],
                           [-1, 5.5, -1],
                           [0, -1, 0]], np.float32)
        img = cv.filter2D(img, -1, kernel)
        return img

    # Helper function to increase contrast of an image
    def increase_contrast(self, img):
        lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        l, a, b = cv.split(lab_img)
        clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img = cv.merge((cl, a, b))
        img = cv.cvtColor(img, cv.COLOR_LAB2RGB)
        return img

    # Helper function to crop the image and eliminate the borders
    def crop_image(self, image):
        mask = image > 0
        coords = np.argwhere(mask)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        image = image[x0 + 200: x1 - 20, y0: y1]
        return image

    # Another helper function to crop and remove the borders
    def crop_image_v2(self, image, tolerance=0):
        mask = image > tolerance
        image = image[np.ix_(mask.any(1), mask.any(0))]
        return image

    # Helper function to distinguish different ECG signals on specific image
    def separate_components(self, image):
        ret, labels = cv.connectedComponents(image, connectivity=8)
        print(type(labels))

        # mapping component labels to hue value
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
        labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)

        # set background label to white
        labeled_image[label_hue == 0] = 255
        return labeled_image

    # Helper function to display segmented ECG picture
    def display_segments(self, name, item, axis='off'):
        plt.figure(figsize=(12, 9))
        plt.imshow(item)
        plt.title(name)
        plt.axis(axis)
        plt.subplots_adjust(wspace=.05, left=.01, bottom=.01, right=.99, top=.9)
        plt.show()


# Helper function to detect characters
def ocr(image):
    text = pytesseract.image_to_string(image, lang='eng')
    return text


def main():
    digitizer = ECGdigitizer()
    image_name = 'images/test4.jpeg'  # select image
    image = cv.imread(image_name, flags=cv.IMREAD_GRAYSCALE)  # read the image as GS

    # sanity check
    if image is None:
        print('Cannot open image: ' + image_name)
        sys.exit(0)
    digitizer.display_image(image, 'Original Image')

    # crop out upper region
    cropped_image = digitizer.crop_image(image)
    digitizer.display_image(cropped_image, 'CROPPED')

    # use thresholding to transform the image into a binary one
    ret, binary_image = cv.threshold(cropped_image, 127, 255, cv.THRESH_BINARY)
    digitizer.display_image(binary_image, 'Binary Image')

    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)
    labels, nb = ndimage.label(binary_image, structure=structure)
    digitizer.display_segments('Labeled Image', labels)

    print('There are ' + str(np.amax(labels) + 1) + ' labeled components.')

    curve_indices = []
    fig = plt.figure(figsize=(12, 8))
    plt.title('Separated Curves')
    columns = 1
    rows = 5
    for i in range(1, np.amax(labels) + 1):
        sl = ndimage.find_objects(labels == i)
        img = binary_image[sl[0]]
        if img.shape[1] > 200:
            curve_indices.append(i)
            fig.add_subplot(rows, columns, len(curve_indices))
            plt.imshow(img, cmap='gray')
        else:
            continue
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    columns = 10
    rows = 5
    s_indices = []
    for i in range(1, np.amax(labels) + 1):
        sl = ndimage.find_objects(labels == i)
        img = binary_image[sl[0]]
        if 10 < img.shape[0] < 12 and 6 < img.shape[1] < 8:
            print(img.shape)
            s_indices.append(i)
            fig.add_subplot(rows, columns, len(s_indices))
            plt.imshow(img, cmap='gray')
        else:
            continue
    plt.show()


if __name__ == '__main__':
    main()
