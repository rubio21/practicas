import cv2

cap = cv2.VideoCapture('sev_ala.mp4')

start=24400
stop=188000
x=0
while True:
	cap.set(cv2.CAP_PROP_POS_FRAMES, start)
	success, img = cap.read()
	# cv2.imshow('a', img)
	# cv2.waitKey()
	cv2.imwrite('frames/sev_ala/frame_'+str(x)+'.png', img)
	x+=1
	start+=500
	if start>=stop:
		break