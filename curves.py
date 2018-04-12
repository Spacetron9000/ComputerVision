#Kyle M. Medeiros
#CAP 4410
#2/15/2018

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def nothing(*arg):
	pass

def track(wName):
	trackName1 = "Contrast"
	trackName2 = "Brightness"
	cv2.namedWindow(wName)
	cv2.createTrackbar(trackName1, wName, 100,300,nothing)
	cv2.createTrackbar(trackName2, wName, 0,100,nothing)
	cap = cv2.VideoCapture('barriers.avi')
	frame_count = 0
	numImagesMade = 0
	#main loop
	while True:
		ret,frame = cap.read()
		frame_count+=1

		#Process Trackbars
		trackPos1 = cv2.getTrackbarPos(trackName1, wName)/100
		trackPos2 = int(cv2.getTrackbarPos(trackName2, wName))-50

		black_img = np.zeros(frame.shape, np.uint8)


		contmask = cv2.add(black_img, np.array([float(trackPos1)]))

		brightmask = cv2.add(black_img, np.array([float(trackPos2)]))
		mult_img= cv2.multiply(frame,contmask)
		result = cv2.add(mult_img, brightmask)


		#This will reset the video loop when the last frame is reached
		#and also save the last frame of the video as an output image
		if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			numImagesMade+=1
			frame_count = 0
			cv2.imwrite('output'+str(numImagesMade)+'.png',result)
			cap.set(cv2.CAP_PROP_POS_FRAMES,0)


		#Display a histogram that continuously updates per each frame
		'''
		hist_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
		hist = cv2.calcHist([hist_img], [0], None, [256], [0, 256])
		fig = plt.figure()
		plt.title("grayscale histogram")
		plt.plot(hist)
		plt.xlim([0, 256])

		#uncomment this in order to see a histogram of the frame plotted
		#each time, however it will essentially disable the live video updating
		#as plt.show() waits for a user to exit the plot.
		#plt.show()


		'''
		cv2.imshow(wName, result)

		ch = cv2.waitKey(5)
		if(ch == 27):
			break

	cv2.destroyAllWindows()
	return 1


track("window")
