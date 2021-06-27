from imutils import contours
import numpy as np
import imutils
import cv2
import os
import time
import traceback
import scan_utils as su

debug_mode = True
su.mkdir_base_folder()

def test_scan(path_img):
	input_img = cv2.imread(path_img)
	img_height, img_width, img_channels = input_img.shape
	max_weight=1807
	max_heigh=2555
	crop_sbd_position = (int(951 / max_weight* img_width), int(254 / max_heigh * img_height), int(1430 / max_weight* img_width), int(821 / max_heigh * img_height))
	crop_sbd_img = input_img[crop_sbd_position[1]:crop_sbd_position[3], crop_sbd_position[0]:crop_sbd_position[2]]
	
	su.export_img('crop_sbd_img.png', crop_sbd_img)
	#sbd = get_sbd(crop_sbd_img)
	#print ("SBD: " + sbd)
	crop_1_30_position = (
	    int(41 / max_weight* img_width), int(833 / max_heigh * img_height), int(480 / max_weight* img_width),
	    int(2470 / max_heigh * img_height))
	crop_1_30_img = input_img[crop_1_30_position[1]:crop_1_30_position[3], crop_1_30_position[0]:crop_1_30_position[2]]
	ans_1_30 = get_ans('1_30', input_img)

	ans_block_img = get_ans_block(input_img)
	su.export_img('ans_block_img.png', ans_block_img)
	gray_img = cv2.cvtColor(ans_block_img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
	edged_img = cv2.Canny(blurred, 0, 200)
	hsv_img = cv2.cvtColor(ans_block_img, cv2.COLOR_BGR2HSV)
	hsv_img[...,1] = hsv_img[...,1]*2.2
	low_blue = np.array([80, 50, 60])
	high_blue = np.array([128, 255, 255])
	back = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	su.export_img('changed.png', back)
	blue_mask = cv2.inRange(hsv_img, low_blue, high_blue)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	dilate = cv2.dilate(blue_mask, kernel, iterations=2)
	su.export_img('blue_mask.png', dilate)

	th3 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 49, 8)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	dilate_th3= cv2.dilate(th3, kernel, iterations=2)
	thresh = ~dilate_th3
	thresh2s = ~dilate_th3&~dilate
	su.export_img('ans_thresh2.png', thresh)
	su.export_img('ans_thresh2s.png', thresh2s)

	ans_h, ans_w, ans_c = ans_block_img.shape
	penaty_h = 5
	size_1_square = int (ans_h / 6)
	ans = [[]]
	ans_index = ['A', 'B', 'C', 'D']
	# for k in range(0, 6):
	# 	square = thresh[int((k * size_1_square)):int(((k + 1) * size_1_square)), 0:ans_w]
	# 	t_h, t_w = square.shape
	# 	square = square[penaty_h:(t_h - penaty_h), 0:t_w]
	# 	su.export_img('ans_square_' + str(k) + '.png', square)

	# 	size_1_row = (t_h - penaty_h*2) / 5
	# 	size_1_col = (t_w / 4)

	# 	for i in range(0, 5):
	# 		ans.append([])
	# 		index = (k*5 + (i + 1))
	# 		for j in range(0, 4):
	# 			unit_ans = square[int((i * size_1_row)):int(((i + 1) * size_1_row)), int((j * size_1_col)):int(((j + 1) * size_1_col))]
	# 			su.export_img('ans_unit_square_' + str(index) + '_' +  str(j) + '.png', unit_ans)
	# 			percent_non_zero = su.get_percent_non_zero(unit_ans)
	# 			su.debug_print(str(index) + "_" + str(j) + " with percent: " + str(percent_non_zero))
	# 			if percent_non_zero > 13:
	# 				ans[index].append(ans_index[j])
	cnts = cv2.findContours(edged_img.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	su.export_img_cnt('ans_cnt11.png', ans_block_img, cnts, True)

	questionCnts = []
	xCnts= []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		if x > 0 and w * h > 1200 and w >= ans_w / 18 and h >= ans_h / 60 and ar >= 0.7 and ar <= 1.3 and w < ans_w / 2 and h < ans_h / 2:
			#print ("x = " + str(x) + ", y = " + str(y) + ", w = " + str(w) + ", h = " + str(h))
			print (str(w * h))
			questionCnts.append(c)
			if x == 0:
				xCnts.append(c)

	su.export_img_cnt('ans_cnt.png', ans_block_img, questionCnts, True)
	su.debug_print('len cnt: ' + str(len(questionCnts)))
	su.debug_print("[[], ['A'], ['B'], ['B'], ['C'], ['D'], ['B'], ['A'], ['C'], ['D'], ['B'], ['A'], ['B'], [], ['C'], ['A'], ['D'], ['C'], ['A'], ['A'], ['A'], ['C'], ['C'], ['C'], ['B'], ['C'], ['D'], ['C'], ['D'], ['D'], ['D']]")

def get_block_points(input_img):
	img_height, img_width, img_channels = input_img.shape
	right_border_from = int(img_width - img_width/15)
	bottom_border_from = img_height - 250

	crop_border_right = input_img[0:img_height, right_border_from:(img_width - 10)]
	crop_border_bottom = input_img[bottom_border_from:(img_height - 10), 0:img_width]
	su.export_img('crop_border_right.png', crop_border_right)
	su.export_img('crop_border_bottom.png', crop_border_bottom)

	gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	# remove noise by blur image
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	# apply canny edge detection algorithm
	#img_canny = cv2.Canny(blurred, 0, 300)
	edged_img = cv2.Canny(blurred, 0, 200)
	thresh = cv2.adaptiveThreshold(blurred, maxValue=255,
								   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
								   thresholdType=cv2.THRESH_BINARY_INV,
								   blockSize=15,
								   C=8)
	
	su.export_img('block_edged_img.png', edged_img)
	su.export_img('block_thresh.png', thresh)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	block_cnts = []
	for cnt in cnts:
		(x, y, w, h) = cv2.boundingRect(cnt)
		ar = w / float(h)
		area = cv2.contourArea(cnt)
		if (x > right_border_from or y > bottom_border_from) and ar > 1.5 and ar < 4 and (x + w) < (img_width - 10) and (y + h) < (img_height - 10):
			print (str(area) + ", w=" + str(w) + ", h=" + str(h))
			block_cnts.append(cnt)

	su.export_img_cnt('block_cnt.png', input_img, block_cnts, True)
	size_block = len(block_cnts)
	su.debug_print('size_block: ' + str(size_block))

	if size_block != 59:
		block_cnts = []
		for cnt in cnts:
			(x, y, w, h) = cv2.boundingRect(cnt)
			ar = w / float(h)
			area = cv2.contourArea(cnt)
			if area > 300 and (x > right_border_from or y > bottom_border_from) and ar > 1.5 and ar < 4 and (x + w) < (img_width - 10) and (y + h) < (img_height - 10):
				print (area)
				block_cnts.append(cnt)
		size_block = len(block_cnts)	

	if size_block != 59:
		raise Exception('find size block error: ' + str(size_block))

	block_cnts = contours.sort_contours(block_cnts,
		method="top-to-bottom")[0]
	block_cnts_col = block_cnts[:41]
	block_cnts_row = contours.sort_contours(block_cnts[41:],
		method="left-to-right")[0]

	su.export_img_cnt('block_cnt_col_sorted.png', input_img, block_cnts_col, True)
	su.export_img_cnt('block_cnt_row_sorted.png', input_img, block_cnts_row, True)

	last_all_col_x, last_all_col_y, last_all_col_w, last_all_col_h = cv2.boundingRect(block_cnts_col[40])
	first_row_x, first_row_y, first_row_w, first_row_h = cv2.boundingRect(block_cnts_row[1])
	last_row_x, last_row_y, last_row_w, last_row_h = cv2.boundingRect(block_cnts_row[4])
	first_col_x, first_col_y, first_col_w, first_col_h = cv2.boundingRect(block_cnts_col[11])
	last_col_x, last_col_y, last_col_w, last_col_h = cv2.boundingRect(block_cnts_row[17])

	first_row_block = (first_row_x, first_row_y)
	last_row_block = (last_row_x, last_row_y)
	first_col_block = (first_col_x, first_col_y)
	last_col_block = (last_col_x, last_col_y)

	print ('first_row_block: ' + str(block_cnts_row[1]))
	print ('last_row_block: ' + str(last_row_block))
	print ('first_col_block: ' + str(first_col_block))
	print ('last_col_block: ' + str(last_col_block))

	su.export_img_cnt('block_cnt_col_sorted_0.png', input_img, block_cnts_col[10:11], True)
	su.export_img_cnt('block_cnt_col_sorted_40.png', input_img, block_cnts_col[39:40], True)
	su.export_img_cnt('block_cnt_row_sorted_0.png', input_img, block_cnts_row[1:2], True)
	su.export_img_cnt('block_cnt_row_sorted_17.png', input_img, block_cnts_row[16:17], True)

	penaty = 20
	top_left = (first_row_x - penaty, first_col_y)
	top_right = (last_row_x + last_row_w + penaty, first_col_y - penaty)
	bottom_right = (last_row_x + last_row_w + penaty, last_row_y + penaty)
	bottom_left = (first_row_x - penaty, first_row_y + penaty)

	su.export_img('block_befoer.png', input_img[top_left[1]:bottom_left[1], top_left[0]:top_right[0]])
	return su.order_points(np.array([top_left, top_right, bottom_right, bottom_left]))


def get_ans_block(img):
	pts = get_block_points(img)
	warped = su.four_point_transform(img, pts)
	h,w,c = warped.shape
	print ('shape: ' + str(warped.shape))
	return warped[0:(h - 60),0:w]

def get_ans(name, input_img):
	#input_img = su.get_bigest_frame(name, input_img)
	su.export_img(name + '_ans_input.png', input_img)
	gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	# remove noise by blur image
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	# apply canny edge detection algorithm
	#img_canny = cv2.Canny(blurred, 0, 300)
	thresh = cv2.adaptiveThreshold(blurred, maxValue=255,
								   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
								   thresholdType=cv2.THRESH_BINARY_INV,
								   blockSize=15,
								   C=8)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)


	su.export_img_cnt('get_ans_cnt.png', input_img, cnts, False)
	return ''

def get_sbd(input_img):
	input_img = su.get_bigest_frame('sbdinput', input_img)
	su.export_img('sbd_input.png', input_img)

	gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
	hsv_img[...,1] = hsv_img[...,1]*2.2
	low_blue = np.array([80, 50, 60])
	high_blue = np.array([128, 255, 255])
	back = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	su.export_img('changed.png', back)
	blue_mask = cv2.inRange(hsv_img, low_blue, high_blue)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	dilate = cv2.dilate(blue_mask, kernel, iterations=2)
	su.export_img('blue_mask.png', dilate)

	th3 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 49, 8)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	dilate_th3= cv2.dilate(th3, kernel, iterations=2)
	thresh = ~dilate_th3
	thresh2s = ~dilate_th3&~dilate
	su.export_img('thresh2.png', thresh)
	su.export_img('thresh2s.png', thresh2s)

	img_height, img_width, img_channels = input_img.shape
	size_1_col = int(img_width / 10)
	size_1_row = int(img_height / 10)
	sbd = list("----------")
    
	for i in range(0, 10):
		for j in range(0, 10):
			square = thresh2s[int((i * size_1_row)):int(((i + 1) * size_1_row)), int((j * size_1_col)):int(((j + 1) * size_1_col))]
			su.export_img('square_' + str(i) + '_' + str(j) + '.png', square)
			# mask = np.zeros(square.shape, dtype="uint8")
			# cv2.drawContours(mask, , -1, 255, -1)
			# mask = cv2.bitwise_and(square, square, mask=mask)
			# total = cv2.countNonZero(mask)
			#cnts_square = cv2.findContours(square, cv2.RETR_LIST,
			#			cv2.CHAIN_APPROX_SIMPLE)
			#cnts_square = imutils.grab_contours(cnts_square)
			su.export_img('cnt_square_' + str(i) + '_' + str(j) + '.png', square)
			percent_non_zero = su.get_percent_non_zero(square)
			su.debug_print(str(i) + "_" + str(j) + " with percent: " + str(percent_non_zero))
			if percent_non_zero > 8:
				sbd[j] = str(i)


	return ''.join(sbd)

path_img = 'D:\\kt1.png'
test_scan(path_img)