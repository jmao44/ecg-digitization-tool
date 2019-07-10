import sys
import cv2 as cv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


# Helper function to help display an oversized image
def display_image(image, name):
    small_image = cv.resize(image, (0, 0), fx=0.85, fy=0.85)
    cv.imshow(name, small_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Helper function to sharpen the image
def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5.5, -1],
                       [0, -1, 0]], np.float32)
    img = cv.filter2D(img, -1, kernel)
    return img


# Helper function to increase contrast of an image
def increase_contrast(img):
    lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab_img)
    clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv.merge((cl, a, b))
    img = cv.cvtColor(img, cv.COLOR_LAB2RGB)
    return img


# Helper function to distinguish different ECG signals on specific image
def separate_components(image):
    ret, labels = cv.connectedComponents(image, connectivity=8)

    # mapping component labels to hue value
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)

    # set background label to white
    labeled_image[label_hue == 0] = 255
    return labeled_image


image_name = 'images/test.jpg'  # select image
image = cv.imread(image_name)  # read the image

# sanity check
if image is None:
    print('Cannot open image: ' + image_name)
    sys.exit(0)
display_image(image, 'Original Image')

# grayscale check
if len(image.shape) != 2:
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# blur the image to get rid of noise
blurred_image = cv.GaussianBlur(gray_image, (3, 3), 0)
blurred_image = cv.medianBlur(blurred_image, 3)

# apply adaptive threshold to transform to a binary image
binary_image = cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 101, 50)
binary_image_inverted = cv.bitwise_not(binary_image)

kernel = np.ones((4, 4), np.uint8)
dilated_image = cv.dilate(binary_image_inverted, kernel, iterations=1)
kernel = np.ones((4, 4), np.uint8)
eroded_image = cv.erode(dilated_image, kernel, iterations=1)

display_image(binary_image_inverted, 'Processed Image')
display_image(eroded_image, 'Eroded Image')

element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
done = False
skel = np.zeros(binary_image_inverted.shape, np.uint8)
while not done:
    eroded = cv.erode(eroded_image, element)
    temp = cv.dilate(eroded, element)
    temp = cv.subtract(binary_image_inverted, temp)
    skel = cv.bitwise_or(skel, temp)
    binary_image_inverted = eroded.copy()
    zeros = np.size(binary_image_inverted) - cv.countNonZero(binary_image_inverted)
    if zeros == np.size(binary_image_inverted):
        done = True
display_image(skel, 'SKEL')


# display the segmented image
# labeled_image = separate_components(eroded_image)
# display_image(labeled_image, 'Labeled Image')

# cv.imwrite('result_image.png', binary_image)