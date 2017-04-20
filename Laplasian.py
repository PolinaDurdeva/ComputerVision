'''
Created on Mar 30, 2017

@author: polina
'''

import cv2
import numpy as np


n = 3
lap_mask = np.zeros((n,n))
#generate mask
lap_mask[0:] = [1.0,1.0,1.0]
lap_mask[1:] = [1.0,-8.0,1.0]
lap_mask[2:] = [1.0,1.0,1.0]
#laplasian mask (other version)
# lap_mask[0:] = [0.0,1.0,0.0]
# lap_mask[1:] = [1.0,-4.0,1.0]
# lap_mask[2:] = [0.0,1.0,0.0]
maskx = np.zeros((n,n))
masky = np.zeros((n,n))
maskx[0:] = [-1.0,-2.0,-1.0]
maskx[1:] = [0.0,0.0,0.0]
maskx[2:] = [1.0,2.0,1.0]
masky = np.transpose(maskx)


def get_laplasian(img):
    return cv2.filter2D(img, -1, lap_mask)
    
def laplas_correction (img):
    l = cv2.filter2D(img, -1, lap_mask)
    c = -1.0
    return scale(img + (c * l))

    
def avg_correction(img, k):
    avg_mask = np.ones([k,k]) * (1.0 / (k**2))
    return np.uint8(cv2.filter2D(img, -1, avg_mask))
    
def grad_correction(img):
    grad_img = np.abs(img/255.0 - 0.5)
    return np.uint8(grad_img * 255)
    
def rise_correction(image, A=1.5):
    mask = lap_mask
    mask[1,1] = mask[1,1] + A
    return cv2.filter2D(image, -1, mask)

def get_sobol (image):
    g_imgx = cv2.filter2D(image/255.0, -1, maskx)
    g_imgy = cv2.filter2D(image/255.0, -1, masky)
    return np.uint8((np.abs(g_imgx) + np.abs(g_imgy))*255)

def gamma_correction(img, g):
    r = img / 255.0
    r = cv2.pow(r,g)
    return np.uint8( r * 255)
    
def sobol_correction(img):
    return img + get_sobol(img)

def scale(img):
    img=img-np.amin(img)
    img=255*(img*1.0/np.amax(img))
    return img

def product(img1, img2):
    img1 = img1/255.0
    img2 = img2/255.0
    prod = img1 * img2
    return np.uint8(prod*255)

def sum_imges(img1, img2):
    img1 = img1/255.0
    img2 = img2/255.0
    s = img1 + img2
    return np.uint8(s*255)
    

def combine_correction(img, k=3 , g=0.5):
    #laplasian
    lap = get_laplasian(img)
    cv2.imwrite("img/mix/lapl.jpg", lap)
    # image + laplasian
    lap_img = laplas_correction(img)
    cv2.imwrite("img/mix/apply_lap.jpg", lap_img)
    # gradient
    g_xy = get_sobol(img)
    cv2.imwrite("img/mix/apply_sob.jpg", g_xy)
    #smoothing gradient
    avg_img =  avg_correction(g_xy, 5)
    cv2.imwrite("img/mix/avg_filter.jpg", avg_img)
    
    multy_img = scale(product(lap_img, avg_img))
    cv2.imwrite("img/mix/multy_filter.jpg", multy_img)
    # sum original image and product 
    sum_img = scale(img + multy_img)
    cv2.imwrite("img/mix/sum_filter.jpg", sum_img)
    #gamma correction
    gamma_img = gamma_correction(sum_img, g)
    cv2.imwrite("img/mix/gamma_filter.jpg", gamma_img)
    
    mix = np.hstack([img, lap, lap_img, g_xy, avg_img, multy_img, sum_img, gamma_img])
    cv2.imwrite("img/mix/all.jpg",mix)
    result = np.hstack([img, gamma_img])
    cv2.imwrite("img/mix/result.jpg",result)
    
    print "COMB is DONE"
    
    

img = cv2.imread('flag3.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.GaussianBlur(img, ksize = (5,5), sigmaX = 6)poisson
cv2.imwrite("img/mix/origin.jpg", img)
#lap_img = laplas_correction(img)
combine_correction(img, g = 1.2)
#cv2.imwrite("img/lap.jpg", lap_img)
print "Program is over"