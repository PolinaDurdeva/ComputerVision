'''
Created on Apr 13, 2017

@author: polina
'''
import numpy as np
import cv2
from math import ceil



def direct_fourie(img):
    return np.fft.fftshift(np.fft.fft2(img))

def reverse_fourie(f_ishift):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(f_ishift)))

def get_mag_spectrum(f_ishift):
    return 20 * np.log(np.abs(f_ishift)+1)

def cut_square(img, x,y):
    rows, cols = img.shape
    crow,ccol = ceil(rows/2) , ceil(cols/2+1)
    res_img = np.copy(img) 
    res_img[crow - y : crow + y + 1, ccol - x : ccol + x + 1] = 0
    return res_img

def cut_diag(img, h):
    res_img = np.copy(img) 
    a = img.shape[0]
    b = img.shape[1]
    k =  a * 1.0 / b
    for i in range(a):
        for j in range(b):
            if ( i == ceil(j * k) or (i <= ceil(j * k) + h and i >= ceil(j * k)) or (i >= ceil(j * k) - h and i <= ceil(j * k))):
                res_img[i][j] = 0
                res_img[i][b - j-1] = 0
    return res_img

def cut_external_square(img, x,y):
    rows, cols = img.shape
    crow,ccol = ceil(rows/2) , ceil(cols/2+1)
    res_img = np.ones(img.shape,dtype=np.complex64)
    inner_img =  img[crow - y : crow + y + 1, ccol - x : ccol + x + 1]
    res_img[crow - y : crow + y + 1, ccol - x : ccol + x + 1] = inner_img
    return res_img

img = cv2.imread('ss.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite("img/Fourie/origin.jpg", img)

df = direct_fourie(img)
cv2.imwrite("img/Fourie/fourie.jpg", np.log(np.abs(df)+1))

df_mag = get_mag_spectrum(df)
cv2.imwrite("img/Fourie/log_furie.jpg", df_mag)

rf = reverse_fourie(df)
cv2.imwrite("img/Fourie/reverse_fourie.jpg", rf)
#######

img_h = cut_square(df, 10, 10)
cv2.imwrite("img/Fourie/img_sq.jpg", np.log(np.abs(img_h) + 1))

df_mag = get_mag_spectrum(img_h)
cv2.imwrite("img/Fourie/log_furie_h.jpg", df_mag)

rf = reverse_fourie(img_h)
cv2.imwrite("img/Fourie/reverse_fourie_h.jpg", rf)

#######

img_d = cut_diag(df, 10)
cv2.imwrite("img/Fourie/img_r.jpg", np.log(np.abs(img_d) + 1))

df_mag = get_mag_spectrum(img_d)
cv2.imwrite("img/Fourie/log_furie_r.jpg", df_mag)

rf = reverse_fourie(img_d)
cv2.imwrite("img/Fourie/reverse_fourie_r.jpg", rf)

#######

img_r = cut_external_square(df, 17, 17)
cv2.imwrite("img/Fourie/img_r.jpg", np.log(np.abs(img_r) + 1))

df_mag = get_mag_spectrum(img_r)
cv2.imwrite("img/Fourie/log_furie_r.jpg", df_mag)

rf = reverse_fourie(img_r)
cv2.imwrite("img/Fourie/reverse_fourie_r.jpg", rf)

print "done"

