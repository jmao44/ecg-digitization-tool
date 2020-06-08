import numpy as np
import cv2 as cv
from os import listdir
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage

from utils import display_segments, display_image

STRUCTURE = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]], np.uint8)

root_directory = '../chongqing'
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
        # print('Original image shape: {}'.format(image.shape))

        cropped_image = image[150:800, 60:900]
        cropped_image_with_header = image[150:800, 0:50]
        # display_image(cropped_image, 'Cropped Image')
        # print('Cropped image shape: {}'.format(cropped_image.shape))

        ret, binary_image = cv.threshold(cropped_image, 50, 255, cv.THRESH_BINARY)
        ret, binary_image_with_header = cv.threshold(cropped_image_with_header, 50, 255, cv.THRESH_BINARY)
        # display_image(binary_image, 'Binary Image')

        ret, labels = cv.connectedComponents(binary_image)
        N = 100
        for i in range(1, labels.max() + 1):
            pts = np.where(labels == i)
            if len(pts[0]) < N:
                labels[pts] = 0
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_image = cv.merge([label_hue, blank_ch, blank_ch])
        labeled_image = cv.cvtColor(labeled_image, cv.COLOR_HSV2BGR)
        labeled_image[label_hue == 0] = 0
        # display_image(labeled_image, 'Labeled Image')

        labeled_image = cv.cvtColor(labeled_image, cv.COLOR_BGR2GRAY)
        labeled_image, cc_num = ndimage.label(labeled_image, structure=STRUCTURE)
        # display_segments(labeled_image, 'Labeled Image')

        curve_indices = []
        curve_lengths = []
        curve_widths = []
        curve_lower_bounds = []
        curve_upper_bounds = []

        # fig = plt.figure(figsize=(12, 9))
        # plt.title('Separated Curves')
        # columns = 1
        # rows = 5
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

        # display_image(binary_image_with_header, 'BINARY IMAGE FULL')

        baselines = [90, 209, 328, 447, 566]
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
        new_img.save('../chongqing_results/' + person_name + '/' + image_name, 'JPEG')







