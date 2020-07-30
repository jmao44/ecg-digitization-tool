import numpy as np
import cv2 as cv
from os import listdir, getcwd
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt

from utils import display_image, display_segments

STRUCTURE = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]], np.uint8)

invalid_pics = []
root_directory = '../6'
for person_name in listdir(root_directory):
    if person_name == '.DS_Store':
        continue
    person_dir = root_directory + '/' + person_name
    for image_name in listdir(person_dir):
        if image_name == '.DS_Store':
            continue
        image_path = person_dir + '/' + image_name
        image = cv.imread(image_path, flags=cv.IMREAD_GRAYSCALE)
        print('Processing: {}'.format(image_path))
        # display_image(image, 'Original Image')

        # image is of type: numpy.ndarray
        # print('Original image shape: {}'.format(image.shape))

        # cropping image
        cropped_image = image[250:900, 100:1500]
        cropped_image_with_header = image[250:900, 0:1500]
        # display_image(cropped_image, 'Cropped Image')
        # print('Cropped image shape: {}'.format(cropped_image.shape))

        # image thresholding
        ret, binary_image = cv.threshold(cropped_image, 127, 255, cv.THRESH_BINARY)
        ret_with_header, binary_image_with_header = cv.threshold(cropped_image_with_header, 127, 255, cv.THRESH_BINARY)
        # display_image(binary_image, 'Binary Image')

        # image segmenting
        labeled_image, nb = ndimage.label(binary_image, structure=STRUCTURE)
        labeled_image_with_header, nb_with_header = ndimage.label(binary_image_with_header, structure=STRUCTURE)
        # display_segments(labeled_image, 'Labeled Image')
        # print('There are ' + str(np.amax(labeled_image)) + ' labeled components. (background not included)')

        curve_indices = []
        curve_lengths = []
        curve_widths = []
        curve_lower_bounds = []
        curve_upper_bounds = []

        # fig = plt.figure(figsize=(12, 9))
        # plt.title('Separated Curves')
        # columns = 1
        # rows = 6
        for i in range(1, np.amax(labeled_image) + 1):
            sl = ndimage.find_objects(labeled_image == i)
            img = binary_image[sl[0]]
            if img.shape[1] > 200:
                curve_indices.append(i)
                curve_widths.append(img.shape[0])
                curve_lengths.append(img.shape[1])
                curve_lower_bounds.append(sl[0][0].stop)
                curve_upper_bounds.append(sl[0][0].start)
                # print("Curve {} line range = [{}, {}].".format(len(curve_indices), sl[0][0].start, sl[0][0].stop))
                # fig.add_subplot(rows, columns, len(curve_indices))
                # plt.imshow(img, cmap='gray')
            else:
                continue
        # plt.show()

        # print()
        # print("Effective curves are components from indices: ", curve_indices)
        # print("Their corresponding curve lengths are: ", curve_lengths)
        # print("Their corresponding curve widths are: ", curve_widths)
        # print()

        # fig = plt.figure(figsize=(12, 9))
        # plt.title("Extracted 'S'")
        # columns = 5
        # rows = 2

        # for recording the baselines of the curves
        baselines = []
        for i in range(1, np.amax(labeled_image_with_header) + 1):
            sl = ndimage.find_objects(labeled_image_with_header == i)
            img = binary_image_with_header[sl[0]]
            if 10 < img.shape[0] < 12 and 6 < img.shape[1] < 8:
                if len(baselines) == 5:
                    break
                baselines.append(sl[0][0].stop)
                # print("'S' {} line range = [{}, {}].".format(len(baselines), sl[0][0].start, sl[0][0].stop))
                # fig.add_subplot(rows, columns, len(baselines))
                # plt.imshow(img, cmap='gray')
            else:
                continue
        # plt.show()
        # print("The corresponding baselines for the curves are: ", baselines)
        # print()
        #
        # fig = plt.figure(figsize=(12, 8))
        # plt.title('Scattered Dots')
        # columns = 1
        # rows = 5

        coords = []
        for i in range(len(curve_indices)):
            sl = ndimage.find_objects(labeled_image == curve_indices[i])
            curve = binary_image[sl[0]]
            length = curve.shape[1]
            width = curve.shape[0]
            xs = []
            ys = []
            for j in range(length):
                for k in range(width - 1, -1, -1):
                    if curve[k][j] == 255:
                        xs.append(j)
                        ys.append(width - k)
                        break
                    else:
                        continue
            # fig.add_subplot(rows, columns, i + 1)
            coords.append(ys)
            # plt.plot(xs, ys)
        # plt.show()

        bigger_pic = []
        try:
            for i in range(len(baselines)):
                axis = baselines[i]
                gs_img = []
                for j in range(len(coords[0])):
                    actual_coord = curve_upper_bounds[i] + coords[i][j]
                    g = 127 + actual_coord - axis
                    gs_img.append(g)
                bigger_pic.append(gs_img)
            array = np.array(bigger_pic, dtype=np.uint8)
            new_img = Image.fromarray(array)
            # new_img.show()
            new_img.save('../6_results/' + person_name + '/' + image_name, 'JPEG')
        except IndexError as e:
            invalid_pics.append(image_path)
            continue

print()
print('Invalid pictures: {}'.format(invalid_pics))