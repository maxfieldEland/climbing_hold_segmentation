
# import the necessary packages
import imutils
import cv2
import matplotlib.pyplot as plt
# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread("wall_2.jpg")
img = cv2.imread("wall2.jpg")
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# blurring reduces the noise of problem areas
for i in range(50):
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

thresh = cv2.threshold(blurred, 150,cv2.BORDER_REPLICATE, cv2.THRESH_BINARY)[1]
plt.imshow(thresh)
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(resized, cnts, -1, (0, 255, 0), 2)	# show the output image
plt.imshow(resized)


# loop over the contours
# =============================================================================
for idx,c in enumerate(cnts):
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c = c*ratio
	cnts[idx] = c.astype("int")
 
cv2.drawContours(image, cnts, -1, (-1, 255, -1), 13)	# show the output image
plt.imshow(img)
