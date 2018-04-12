
#Kyle M. Medeiros
#CAP 4410
#3/31/2018

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Image input

filename = 'south_L'
img = cv2.imread('Input_Data/'+filename+'-150x150.png')
img = cv2.resize(img, (img.shape[0]*3,img.shape[1]*3))
cv2.imshow("image",img)
cv2.waitKey(0)

#Grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Binary thresholding
ret, thresh = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

mask = np.zeros(img.shape[:2],np.uint8)
#floodmask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
rect = (1,1,img.shape[0] - 1,img.shape[1] -1 )

#Internal arrays for grabcut
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#blur for edge detection
#kernel = 5

#gauss = cv2.GaussianBlur(img, (kernel,kernel),0)

#cv2.floodFill(gauss,floodmask,(0,0),255)

#cv2.imshow("gauss + flood",gauss)
#cv2.waitKey(0)
#edge = cv2.Canny(gauss,50,200)




#edge = cv2.bitwise_not(edge)
#cv2.imshow("edge",edge)


#grabcut mode to try and get a better foreground extraction
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,7,cv2.cv.GC_INIT_WITH_RECT)

#comparison with canny edge (experimental)
#mask[edge==0] = 0
#mask[edge==255] = 1



mask2 = np.where((mask==1)+(mask==3),255,0).astype('uint8')

#Fore will be the rough grabcut image
fore = np.zeros((img.shape[0],img.shape[1]))
fore = cv2.bitwise_and(img,img,mask=mask2)


cv2.imshow("fore",fore)
cv2.waitKey(0)

cv2.imwrite('output_img/'+filename+'_foreground.png',fore)

gray2 = cv2.cvtColor(fore,cv2.COLOR_BGR2GRAY)

#Finding the background and foreground
kernel = np.ones((3,3),np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening, kernel,iterations = 3)
dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)

#Thresholding the grayscale of the foreground
ret, sure_fg = cv2.threshold(gray2,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

sure_fg = np.uint8(sure_fg)

#anything that isn't surely foreground or background is unknown
unknown = cv2.subtract(sure_bg, sure_fg)

preconcat = cv2.hconcat((sure_fg,sure_bg))
watershedconcat = cv2.hconcat((preconcat,unknown))

cv2.imwrite('output_img/'+filename+'_fg_bg_unknown.png',watershedconcat)
'''
cv2.imshow("fg",sure_fg)
cv2.imshow("bg",sure_bg)
cv2.imshow("unknown",unknown)
'''
cv2.imshow("fg,bg,unknown",watershedconcat)

cv2.waitKey(0)
contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

marker = np.zeros((gray.shape[0],gray.shape[1]),dtype = np.int32)
marker = np.int32(sure_fg) + np.int32(sure_bg)

for id in range(len(contours)):
    cv2.drawContours(marker, contours, id, id+2, -1)

marker = marker+1

marker[unknown==255] = 0

cv2.watershed(img,marker)
img[marker==-1] =(0,0,255)
imgplt = plt.imshow(marker)
plt.colorbar()
plt.savefig('output_img/'+filename+'_segmentation.png')
plt.show()

cv2.destroyAllWindows()
