import cv2 as cv


def display_image(image, name):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def hello():
    print('hello')

