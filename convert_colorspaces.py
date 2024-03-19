
from matplotlib import pyplot as plt
import numpy as np
import cv2

amount = 3

rgbImage = []
ycbcrImage = []
customHsvImage = []

for x in range(amount):

    # Read the RGB image
    rgbImage.append(cv2.imread(f'fish{x}.png')[:, :, :3])



    # Apply the conversion matrix and reshape the result back to the image dimensions
    ycbcrImage.append(cv2.cvtColor(rgbImage[x], cv2.COLOR_BGR2YCrCb))


    # Convert RGB to HSV using the function
    # Separate the H, S, and V channels
    image = cv2.cvtColor(rgbImage[x], cv2.COLOR_BGR2HSV)
    customHsvImage.append(np.reshape(image, rgbImage[x].shape))

row = amount
col = 4

# Display:
plt.figure()

for x in range(amount):

    # The converted HSV and YCbCr images

    # The original RGB image
    plt.subplot(row, col, x * col + 1), plt.imshow(rgbImage[x]), plt.title('Original RGB Image' if x == 0 else '')
    plt.subplot(row, col, x * col + 2), plt.imshow(customHsvImage[x]), plt.title('Original RGB Image' if x == 0 else '')
    plt.subplot(row, col, x * col + 3), plt.imshow(ycbcrImage[x][:, :, 1]), plt.title('Original RGB Image' if x == 0 else '')
    plt.subplot(row, col, x * col + 4), plt.imshow(ycbcrImage[x][:, :, 2]), plt.title('Original RGB Image' if x == 0 else '')


plt.show()
