import cv2
import numpy as np

# Load the image
image = cv2.imread('homer.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image 'homer.png' not found.")

# Apply binary thresholding
retval, threshold = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold', threshold)

# Detect edges using Canny edge detector
edges = cv2.Canny(threshold, 100, 200)
cv2.imshow('Edges', edges)

# Convert the original image to HSV color space
# Define the lower and upper bounds for skin

lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")
converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
converted = cv2.cvtColor(converted, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)
cv2.imshow('Skin Mask', skinMask)

# Apply dilation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
dilation = cv2.dilate(image, kernel, iterations=1)
cv2.imshow('Dilation', dilation)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
