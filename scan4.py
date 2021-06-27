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
	#input_img = su.convert_to_binary_img(input_img)
	img_height, img_width = input_img.shape[0:2]
	max_weight=1807
	max_heigh=2555
	crop_sbd_position = (int(951 / max_weight* img_width), int(254 / max_heigh * img_height), int(1430 / max_weight* img_width), int(821 / max_heigh * img_height))
	crop_sbd_img = input_img[crop_sbd_position[1]:crop_sbd_position[3], crop_sbd_position[0]:crop_sbd_position[2]]
	su.export_img('crop_sbd_img.png', crop_sbd_img)
	#sbd = get_sbd(crop_sbd_img)
	#print ("SBD: " + sbd)

	block_cnts = get_block_cnts(input_img)
	ans_block_names = ['', '1_30', '31_60', '61_90', '91_120']

	for col_index in range(1, 5):
		name_block = ans_block_names[col_index]
		ans_block_img = get_ans_block(input_img, block_cnts, col_index)
		su.export_img(name_block + '_ans_block_img.png', ans_block_img)

		ans, cnts_correct = get_ans(name_block, ans_block_img)
		su.export_img_cnt('000_' + name_block + '_final.png', ans_block_img, cnts_correct, False)

	# ans_block_1_30_img = get_ans_block(input_img, block_cnts, 1)
	# su.export_img('1_30_ans_block_img.png', ans_block_1_30_img)

	# ans_1_30, cnts_1_30 = get_ans('1_30', ans_block_1_30_img)
	# su.export_img_cnt('1_30_final.png', ans_block_1_30_img, cnts_1_30, False)

	# ans_block_31_60_img = get_ans_block(input_img, block_cnts, 2)
	# su.export_img('31_60_ans_block_img.png', ans_block_31_60_img)

	# ans_31_60, cnts_31_60 = get_ans('1_30', ans_block_31_60_img)
	# su.export_img_cnt('31_60_final.png', ans_block_31_60_img, cnts_31_60, False)


def find_min_area(cnts):
	min_size = 9999999999999
	for cnt in cnts:
		(x, y, w, h) = cv2.boundingRect(cnt)
		if w * h < min_size:
			min_size = w * h 

	return min_size

def binary_search_question_cnt(cnts):
	left = 0
	right = 5000
	mid = (left + right ) // 2

	while left < right and left != mid and mid != right:
		question_cnts = find_question_cnt(cnts, mid)
		size_question_cnts = len(question_cnts)

		print ("mid = " + str(mid) + ", size = " + str(size_question_cnts))
		if size_question_cnts == 120:
			return question_cnts

		if size_question_cnts < 120:
			right = mid
			mid = (left + right) // 2
		else:
			left = mid 
			mid = (left + right) // 2

	return []

def find_question_cnt(cnts, min_area):
	question_cnts = []
	for cnt in cnts:
		(x, y, w, h) = cv2.boundingRect(cnt)
		ar = w / float(h)
		area = w * h #cv2.contourArea(cnt)
		if (x > 0 or y > 0) and ar > 0.5 and ar < 1.5 and area > min_area:
			#print (str(area) + ", w=" + str(w) + ", h=" + str(h))
			question_cnts.append(cnt)

	return question_cnts

def binary_search_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts):
	left = 0
	right = 5000
	mid = (left + right ) // 2
	expect_size = 59

	while left < right and left != mid and mid != right:
		result_cnts = find_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts, mid)
		size_cnts = len(result_cnts)

		print ("mid = " + str(mid) + ", size = " + str(size_cnts))
		if size_cnts == expect_size:
			return result_cnts

		if size_cnts < expect_size:
			right = mid
			mid = (left + right) // 2
		else:
			left = mid 
			mid = (left + right) // 2

	return []

def find_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts, min_area):
	block_cnts = []
	for cnt in cnts:
		(x, y, w, h) = cv2.boundingRect(cnt)
		ar = w / float(h)
		area = w * h #cv2.contourArea(cnt)
		if area >= min_area and (x > right_border_from or y > bottom_border_from) and ar > 1.5 and ar < 4 and (x + w) < (img_width - 10) and (y + h) < (img_height - 10):
			#print (str(area) + ", w=" + str(w) + ", h=" + str(h))
			block_cnts.append(cnt)

	return block_cnts

def get_ans(name, ans_block_img):
	#su.export_img(name + '.png', input_img)
	ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(ans_block_img)

	su.export_img(name + 'ans_block_binary_img.png', ans_block_binary_img)
	su.export_img(name + 'ans_block_outgray_img.png', ans_block_outgray_img)
	su.export_img(name + 'ans_block_bg_img.png', ans_block_bg_img)
	thresh = cv2.adaptiveThreshold(ans_block_binary_img, maxValue=255,
								   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
								   thresholdType=cv2.THRESH_BINARY_INV,
								   blockSize=15,
								   C=8)

	thresh_bg = cv2.adaptiveThreshold(ans_block_bg_img, maxValue=255,
								   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
								   thresholdType=cv2.THRESH_BINARY_INV,
								   blockSize=15,
								   C=8)

	su.export_img(name + '_thresh.png', thresh)
	su.export_img(name + '_thresh_bg.png', thresh_bg)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	su.export_img_cnt(name + '_cnts.png', ans_block_img, cnts, True)

	question_cnts = binary_search_question_cnt(cnts)
	su.export_img_cnt(name + '_question_cnts.png', ans_block_img, question_cnts, True)

	size_question_cnts = len(question_cnts)
	su.debug_print('size_question_cnts: ' + str(size_question_cnts))

	if size_question_cnts != 120:
		raise Exception('find size_question_cnts error: ' + str(size_question_cnts))

	index_question = 0
	question_cnts = contours.sort_contours(question_cnts,
	method="top-to-bottom")[0]
	results = []
	percents = []
	max_percent = 0
	result_item_text = ['A', 'B', 'C', 'D']

	min_size = find_min_area(question_cnts)
	for (q, i) in enumerate(np.arange(0, size_question_cnts, 4)):
		question_row_cnts = contours.sort_contours(question_cnts[i:(i+4)],
											  method="left-to-right")[0]

		for cnt in question_row_cnts:
			results.append('-')
			(x, y, w, h) = cv2.boundingRect(cnt)
			question_item = thresh_bg[y:(y + h), x:(x + w)]
			#su.export_img(name + '_question' + str(index_question + 1) + '.png', question_item)
			percent_non_zero = su.get_percent_non_zero_with_size(question_item, min_size)
			#non_zero = su.get_percent_non_zero_mask(str(index_question), question_item, ans_block_img[y:(y + h), x:(x + w)])
			#su.debug_print(str(index_question + 1) + " with percent: " + str(percent_non_zero))
			index_question += 1
			percents.append({'value': percent_non_zero, 'cnt': cnt})
			if percent_non_zero > max_percent:
				max_percent = percent_non_zero

	max_percent = 27 if max_percent < 27 else max_percent
	temp = max_percent / 1.7
	min_percent_correct = temp if temp > 23 else 23
	cnts_correct = []
	for (q, i) in enumerate(np.arange(0, size_question_cnts, 4)):
		index_ans = 0
		result_item = ''
		temp_percents = su.sub_list(percents, i, i + 4)
		for percent in temp_percents:
			if percent['value'] > min_percent_correct:
				result_item += result_item_text[index_ans]
				cnts_correct.append(percent['cnt'])
			index_ans+= 1

		results.append(result_item)

	return results, cnts_correct

#
# Get block points and transforms 4 points
#
def get_ans_block(input_img, block_cnts, col_index):
	pts = get_block_points(input_img, block_cnts, col_index)
	warped = su.four_point_transform(input_img, pts)
	h,w,c = warped.shape
	#print ('shape: ' + str(warped.shape))
	return warped[0:(h - 60),0:w]

# 
# Get 59 border around block points
#
def get_block_cnts(input_img):
	img_height, img_width = input_img.shape[0:2]
	right_border_from = int(img_width - img_width/15)
	bottom_border_from = img_height - 200

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
			#print (str(area) + ", w=" + str(w) + ", h=" + str(h))
			block_cnts.append(cnt)

	su.export_img_cnt('block_cnt.png', input_img, block_cnts, True)
	size_block = len(block_cnts)
	su.debug_print('size_block_1: ' + str(size_block))

	if size_block != 59:
		block_cnts = binary_search_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts)
		size_block = len(block_cnts)	

	su.debug_print('size_block_2: ' + str(size_block))
	if size_block != 59:
		raise Exception('find size block error: ' + str(size_block))

	block_cnts = contours.sort_contours(block_cnts,
		method="top-to-bottom")[0]
	su.export_img_cnt('block_cnt2.png', input_img, block_cnts, True)
	return block_cnts

def get_block_points(input_img, block_cnts, col_index):
	block_cnts_col = block_cnts[:41]
	block_cnts_row = contours.sort_contours(block_cnts[41:],
		method="left-to-right")[0]

	su.export_img_cnt('block_cnt_col_sorted.png', input_img, block_cnts_col, True)
	su.export_img_cnt('block_cnt_row_sorted.png', input_img, block_cnts_row, True)

	last_all_col_x, last_all_col_y, last_all_col_w, last_all_col_h = cv2.boundingRect(block_cnts_col[40])
	first_row_x, first_row_y, first_row_w, first_row_h = cv2.boundingRect(block_cnts_row[1 + 4 * (col_index - 1)])
	last_row_x, last_row_y, last_row_w, last_row_h = cv2.boundingRect(block_cnts_row[4 + 4 * (col_index - 1)])
	first_col_x, first_col_y, first_col_w, first_col_h = cv2.boundingRect(block_cnts_col[11])
	last_col_x, last_col_y, last_col_w, last_col_h = cv2.boundingRect(block_cnts_row[17])

	first_row_block = (first_row_x, first_row_y)
	last_row_block = (last_row_x, last_row_y)
	first_col_block = (first_col_x, first_col_y)
	last_col_block = (last_col_x, last_col_y)

	# print ('first_row_block: ' + str(block_cnts_row[1]))
	# print ('last_row_block: ' + str(last_row_block))
	# print ('first_col_block: ' + str(first_col_block))
	# print ('last_col_block: ' + str(last_col_block))

	# su.export_img_cnt('block_cnt_col_sorted_0.png', input_img, block_cnts_col[10:11], True)
	# su.export_img_cnt('block_cnt_col_sorted_40.png', input_img, block_cnts_col[39:40], True)
	# su.export_img_cnt('block_cnt_row_sorted_0.png', input_img, block_cnts_row[1:2], True)
	# su.export_img_cnt('block_cnt_row_sorted_17.png', input_img, block_cnts_row[16:17], True)

	penaty = 40
	top_left = (first_row_x - penaty, first_col_y - penaty)
	top_right = (last_row_x + last_row_w + penaty, first_col_y - penaty)
	bottom_right = (last_row_x + last_row_w + penaty, last_row_y + penaty)
	bottom_left = (first_row_x - penaty, first_row_y + penaty)

	su.export_img(str(col_index) + '_block_before.png', input_img[top_left[1]:bottom_left[1], top_left[0]:top_right[0]])
	return su.order_points(np.array([top_left, top_right, bottom_right, bottom_left]))

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

path_img = 'E:\\hgedu-test\\kt5.png'
test_scan(path_img)