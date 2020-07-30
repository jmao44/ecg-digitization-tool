import cv2 as cv
from matplotlib import pyplot as plt


def display_image(image, name):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def display_segments(image, name, axis='off'):
    plt.figure(figsize=(12, 9))
    plt.imshow(image)
    plt.title(name)
    plt.axis(axis)
    plt.subplots_adjust(wspace=.05, left=.01, bottom=.01, right=.99, top=.9)
    plt.show()

