import numpy as np
import cv2
import matplotlib.pyplot as plt

def equalization(img):
    print "Equalization"
    b = np.zeros(255)
    r = np.zeros(img.shape)
    p_count = 0
    get_hist(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[img[i][j]] += 1
            p_count += 1
    b = b / p_count

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r[i][j] = sum(b[ :img[i][j]])
    get_hist(r*255)        
    return np.uint8(r * 255)


def local_equalization(img, D, H):
    r = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b = np.zeros(255)
            for k in range(i - D/2, i + D/2+1):
                for l in range(j - H/2, j + H/2):
                    if (0 <= k < img.shape[0] and 0 <= l < img.shape[1]):
                        T = img[k][l]
                        b[T] += 1
            b = b / (D * H)
            r[i][j] = sum(b[:img[i][j]])
    get_hist(r*255) 
    return np.uint8(r * 255)

def get_hist(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

image = cv2.imread("snake.jpg", cv2.IMREAD_GRAYSCALE)
equalization_image = equalization(image)
l_eq =  local_equalization(image, 60, 60)

cv2.imwrite("Without_Equalization.jpg", image)

cv2.imwrite("Equalization.jpg", equalization_image)

cv2.imwrite("Local_Equalization.jpg", l_eq)


