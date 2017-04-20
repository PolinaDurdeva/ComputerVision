'''
Created on Mar 23, 2017

@author: polina
'''

import numpy as np
import cv2


def special_mask(n):
    m = np.zeros((n,n))
    c = n/2
    d = c
    for i in range(-d, d + 1):
        for j in range(-d, d + 1):
            m[c + i][c + j] = i**2 + j**2
    total_sum = np.sum(m)
    print m / total_sum
    return m / total_sum

     
n = 7
avg_mask = np.ones((n, n),np.float32)/ (n*n)
special_m = special_mask(n)

image = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)
avg_img     = cv2.filter2D(image, -1, avg_mask)
special_img = cv2.filter2D(image, -1, special_m)
median_img  = cv2.medianBlur(image, n)

cv2.imwrite("real.jpg",  image)
cv2.imwrite("3_avg_img.jpg",  avg_img)
cv2.imwrite("3_special_img.jpg",  special_img)
cv2.imwrite("3_median_img.jpg", median_img)
print "Program is over"
