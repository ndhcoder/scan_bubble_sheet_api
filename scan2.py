from imutils import contours
import numpy as np
import imutils
import cv2
import os
import time
import traceback

def current_milli_time():
	return round(time.time() * 1000)

base_folder = './test_debug/' + str(current_milli_time())
os.mkdir(base_folder)

def export_img(name, export_img):
	cv2.imwrite(base_folder + '/' + name, export_img)

def export_img_cnt(name, input_img, cnts, is_create_new_img):
	export_img = np.zeros(input_img.shape, dtype="int32") if is_create_new_img else input_img
	cv2.drawContours(export_img, cnts, -1, (0, 255, 0), 3)
	cv2.imwrite(base_folder + '/' + name, export_img)
#ok

def test_scan(path_img):
	input_img = cv2.imread(path_img)
	img_height, img_width, img_channels = input_img.shape
	max_weight=1807
	max_heigh=2555
	crop_sbd_position = (int(951 / max_weight* img_width), int(254 / max_heigh * img_height), int(1430 / max_weight* img_width), int(821 / max_heigh * img_height))
	crop_sbd_img = input_img[crop_sbd_position[1]:crop_sbd_position[3], crop_sbd_position[0]:crop_sbd_position[2]]
	
	export_img('crop_sbd_img.png', crop_sbd_img)
	sbd = get_sbd(crop_sbd_img)
	
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def distance(a, b):
	return ((((b[0] - a[0] )**2) + ((b[1]-a[1])**2)))

def get_pts(cnt):
	x,y,w,h = cv2.boundingRect(cnt)
	top_left = (-1,-1)
	top_right = (-1,-1)
	bottom_right = (-1,-1)
	bottom_left = (-1,-1)

	min_top_left = 99999
	min_top_right = 99999
	min_bottom_left = 99999
	min_bottom_right = 99999

	for p in cnt:
		t1 = distance(p[0], (x, y))
		t2 = distance(p[0], (x, y+h))
		t3 = distance(p[0], (x+w, y+h))
		t4 = distance(p[0], (x+w, y))

		if t1 < min_top_left:
			top_left = p[0]
			min_top_left = t1

		if t2 < min_top_right:
			top_right = p[0]
			min_top_right = t2

		if t3 < min_bottom_right:
			bottom_right = p[0]
			min_bottom_right = t3

		if t4 < min_bottom_left:
			bottom_left = p[0]
			min_bottom_left = t4

	return np.array([top_left, top_right, bottom_right, bottom_left])
	
def get_bigest_frame(key, input_img):
	gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
	edged_img = cv2.Canny(blurred_img, 75, 200)
	thresh_img = cv2.adaptiveThreshold(blurred_img, maxValue=255,
								   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
								   thresholdType=cv2.THRESH_BINARY_INV,
								   blockSize=15,
								   C=8)

	export_img(key + '_edged_img.png', edged_img)
	cnts = cv2.findContours(edged_img, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	export_img_cnt(key + '_cnt_input.png', input_img, cnts, True)
	export_img_cnt(key + '_cnt_input_biggest.png', input_img, cnts[0], True)

	pts = get_pts(cnts[0])
	warped = four_point_transform(input_img, pts)
	return warped

def get_percent_gray(img):
	

def get_sbd(input_img):
	input_img = get_bigest_frame('sbdinput', input_img)
	export_img('sbd_input.png', input_img)

	gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
	edged_img = cv2.Canny(blurred_img, 75, 200)
	thresh_img = cv2.adaptiveThreshold(gray_img, maxValue=255,
								   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
								   thresholdType=cv2.THRESH_BINARY_INV,
								   blockSize=15,
								   C=8)

	export_img('edged_img.png', edged_img)
	cnts = cv2.findContours(edged_img, cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	export_img_cnt('sbd_cnts.png', input_img, cnts, True)
	img_height, img_width, img_channels = input_img.shape
	size_1_col = int(img_width / 10)
	size_1_row = int(img_height / 10)
    
	for i in range(0, 10):
		for j in range(0, 10):
			square = edged_img[int((i * size_1_row)):int(((i + 1) * size_1_row)), int((j * size_1_col)):int(((j + 1) * size_1_col))]
			export_img('square_' + str(i) + '_' + str(j) + '.png', square)
			# mask = np.zeros(square.shape, dtype="uint8")
			# cv2.drawContours(mask, , -1, 255, -1)
			# mask = cv2.bitwise_and(square, square, mask=mask)
			# total = cv2.countNonZero(mask)
			cnts_square = cv2.findContours(square, cv2.RETR_LIST,
						cv2.CHAIN_APPROX_SIMPLE)
			cnts_square = imutils.grab_contours(cnts_square)
			export_img_cnt('cnt_square_' + str(i) + '_' + str(j) + '.png', square, cnts_square, True)

	return "-"

path_img = 'D:\\kt3.png'
test_scan(path_img)