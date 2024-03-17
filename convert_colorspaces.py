
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Read the RGB image
rgbImage = cv2.imread('fuck_you.png')[:, :, :3]

# Reshape the RGB image for matrix multiplication
rgbImageReshaped = np.reshape(np.double(rgbImage), (-1, 3)).T

# Convert RGB to YCbCr using matrix operations
conversionMatrix = np.array([[0.299, 0.587, 0.114],
                             [-0.147, -0.289, 0.436],
                             [0.615, -0.515, -0.1]])

offsets = np.array([0, 128, 128]).reshape(-1, 1)

# Apply the conversion matrix and reshape the result back to the image dimensions
image = np.dot(conversionMatrix, rgbImageReshaped) + offsets
customYcbcrImage = np.reshape(image.T, rgbImage.shape)


# Separate the Y, Cb, and Cr channels
yChannel = customYcbcrImage[:, :, 0]
cbChannel = customYcbcrImage[:, :, 1]
crChannel = customYcbcrImage[:, :, 2]


# Convert RGB to HSV using the function
# Separate the H, S, and V channels
image = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
customHsvImage = np.reshape(image, rgbImage.shape)

hue = customHsvImage[:, :, 0]
saturation = customHsvImage[:, :, 1]
value = customHsvImage[:, :, 2]

row = 3
col = 4

# Display:
plt.figure()

# The converted HSV and YCbCr images

plt.subplot(row, col, col + 1), plt.imshow(np.uint8(customYcbcrImage)), plt.title('Converted YCbCr Image')
plt.subplot(row, col, 2 * col + 1), plt.imshow(customHsvImage), plt.title('Converted HSV Image')

# The separated YCbCr channels
plt.subplot(row, col, col + 2), plt.imshow(np.uint8(yChannel)), plt.title('Y Channel')
plt.subplot(row, col, col + 3), plt.imshow(np.uint8(cbChannel)), plt.title('Cb Channel')
plt.subplot(row, col, col + 4), plt.imshow(np.uint8(crChannel)), plt.title('Cr Channel')

# The separated HSV channels
plt.subplot(row, col, 2 * col + 2), plt.imshow(hue), plt.title('Hue')
plt.subplot(row, col, 2 * col + 3), plt.imshow(saturation), plt.title('Saturation')
plt.subplot(row, col, 2 * col + 4), plt.imshow(value), plt.title('Value')

# The original RGB image
plt.subplot(row, col, 2), plt.imshow(rgbImage), plt.title('Original RGB Image')

# The YCbCr image that was converted using the built-in function
(plt.subplot(row, col, 3), plt.imshow(cv2.cvtColor(rgbImage, cv2.COLOR_BGR2YCrCb)),
    plt.title('Converted YCbCr Image using built-in function'))

plt.show()
