from imutils import contours
import numpy as np
import imutils
import cv2
import os
import time
import traceback


def current_milli_time():
	return round(time.time() * 1000)
debug_mode = True
base_folder = './test_debug/' + str(current_milli_time())

def mkdir_base_folder():
	base_folder = './test_debug/' + str(current_milli_time())
	os.mkdir(base_folder)

def convert_to_binary_img(input_img):
	image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
	bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
	out_gray = cv2.divide(image, bg, scale=255)
	out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
	return [out_binary, out_gray, bg]

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
	export_img(key + '_thresh_img.png', thresh_img)
	cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	export_img_cnt(key + '_cnt_input.png', input_img, cnts, True)
	export_img_cnt(key + '_cnt_input_biggest.png', input_img, cnts[0], True)

	pts = get_pts(cnts[0])
	warped = four_point_transform(input_img, pts)
	return warped

def get_percent_non_zero(img):
	height, width = img.shape
	img_size = height * width 
	non_zero_pixel = cv2.countNonZero(img)
	return ((non_zero_pixel * 100.0) / img_size)

def sub_list(ls, start, end):
	results = []

	for i in range(start, end):
		results.append(ls[i])

	return results

def get_percent_non_zero_with_size(img, img_size):
	non_zero_pixel = cv2.countNonZero(img)
	return ((non_zero_pixel * 100.0) / img_size)

# def get_percent_non_zero_mask(name, thresh_img, bg_test):
# 	cnts = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 	h, w = thresh_img.shape
# 	cnts = imutils.grab_contours(cnts)
# 	#print (name + "_get_percent_non_zero_mask: " + str(len(cnts)))
# 	export_img_cnt(name + '_get_percent_non_zero_mask.png', bg_test, cnts, True)
# 	mask = np.zeros(thresh_img.shape, dtype="uint8")
# 	cv2.drawContours(mask, cnts, -1, 255, -1)
# 	mask = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
# 	non_zero_pixel = cv2.countNonZero(mask)

# 	return non_zero_pixel#((non_zero_pixel * 100.0) / (w * h))


def debug_print(msg):
	if debug_mode:
		print (msg)

def export_img(name, export_img):
	if debug_mode:
		cv2.imwrite(base_folder + '/' + name, export_img)

def export_img_cnt(name, input_img, cnts, is_create_new_img):
	if debug_mode:
		export_img = np.zeros(input_img.shape, dtype="int32") if is_create_new_img else input_img
		cv2.drawContours(export_img, cnts, -1, (0, 255, 0), 3)
		cv2.imwrite(base_folder + '/' + name, export_img)
#ok
def removeBlue(input_img):
	gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
	hsv_img[...,1] = hsv_img[...,1]*2.2
	low_blue = np.array([80, 50, 60])
	high_blue = np.array([128, 255, 255])
	back = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	blue_mask = cv2.inRange(hsv_img, low_blue, high_blue)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	dilate = cv2.dilate(blue_mask, kernel, iterations=2)
	th3 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 49, 8)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	dilate_th3= cv2.dilate(th3, kernel, iterations=2)
	thresh = ~dilate_th3
	thresh2s = ~dilate_th3&~dilate

	return thresh2s