import cv2

import numpy as np


def gamma_correction(img, correction):
    print "Gamma"
    result = img / 255.0
    result = cv2.pow(result, correction)
    return np.uint8(result * 255)


def log_correction(img):
    print "Log"
    b = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i][j] = np.log(1.0 + img[i][j] / 255.0)
    return np.uint8(b * 255)

def neg_correction(img):
    print "Negative"
    b = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i][j] = 1.0 - img[i][j] / 255.0
    return np.uint8(b * 255)

def linear_correction(img):
    k1 = 0.5
    b = np.zeros(img.shape)
    print "Linear"
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i][j] = img[i][j] * k1 / 255.0
    return np.uint8(b * 255)

def p_linear_correction(img):
    k1 = 0.5
    k2 = 2.0
    b = np.zeros(img.shape)
    print "Piecewise linear"
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i][j] = img[i][j] / 255.0
            if b[i][j] > 0.95:
                b[i][j] = b[i][j] * k1
            if b[i][j] < 0.2:
                b[i][j] = b[i][j] * k2
    return np.uint8(b * 255)

########################################

image = cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE)
alpha = 0.4


gamma_image = gamma_correction(image, alpha)
log_image = log_correction(image)
neg_image = neg_correction(image)
pl_image = p_linear_correction(image)
l_image = linear_correction(image)

# cv2.imshow("Gamma {}".format(alpha), gamma_image)
# cv2.waitKey()
# cv2.imshow("Logarithm result", log_image)
# cv2.waitKey()
# cv2.imshow("Negative result", neg_image)
# cv2.waitKey()
# cv2.imshow("Piecewise linear result", pl_image)
# cv2.waitKey()
# cv2.imshow("Linear result", l_image)
# cv2.waitKey()
