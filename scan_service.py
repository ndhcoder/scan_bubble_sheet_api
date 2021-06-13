from imutils import contours
import numpy as np
import imutils
import cv2
import os
import time
import traceback

def get_result_trac_nghiem(image_trac_nghiem, ANSWER_KEY, debug):
	translate = {"A": 0, "B": 1, "C": 2, "D": 3}
	revert_translate={0:"A",1:"B",2:"C",3:"D",-1:"N"}
	image = image_trac_nghiem
	height, width, channels = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

	# cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
	# cv2.imshow("cropped", image)
	# cv2.waitKey(0)

	if debug['on']:
		cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
		cv2.imwrite(debug['folder'] +'contours_tn.' + str(current_milli_time()) + '.jpeg', image)

	questionCnts = []
	xCnts= []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		if x > 0 and w >= width / 25 and h >= height / 70 and ar >= 0.7 and ar <= 1.3 and w < width / 2 and h < height / 2:
			#print ("x = " + str(x) + ", y = " + str(y) + ", w = " + str(w) + ", h = " + str(h))
			questionCnts.append(c)
			if x == 0:
				xCnts.append(c)

	questionCnts = contours.sort_contours(questionCnts,
		method="top-to-bottom")[0]

	# cv2.drawContours(image, questionCnts, -1,  (0, 255, 0), 3)
	# cv2.imshow("cropped", thresh)
	# cv2.waitKey(0)

	if debug['on']:
		cv2.drawContours(image, questionCnts, -1, (0, 0, 255), 3)
		cv2.imwrite(debug['folder'] +'contours_tn2_xx.' + str(current_milli_time()) + '.jpeg', image)
		print("len question cnt = " + str(len(questionCnts)))

	if len(questionCnts)!=120:
		thresh = cv2.threshold(gray, 0, 255,
							   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		questionCnts = []
		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			if x > 0 and w >= width / 25 and h >= height / 70 and ar >= 0.7 and ar <= 1.3 and w < width/2 and h < height / 2:
				questionCnts.append(c)

		questionCnts = contours.sort_contours(questionCnts,
											  method="top-to-bottom")[0]


	print("len question = " + str(len(questionCnts)))
	select=[]
	if len(questionCnts) != 120:
		raise Exception('size questions invalid')
	list_min_black = []

	thresh = cv2.threshold(gray, 0, 255,
						   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	min_black = 1000000000
	for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):

		cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
		for (j, c) in enumerate(cnts):
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			mask = cv2.bitwise_and(thresh, thresh, mask=mask)
			total = cv2.countNonZero(mask)

			# print('total ' + str(total))
			if total <= min_black:
				min_black=total
		# print(i,min_black)
		if (i+4)%20==0:
			list_min_black.append(min_black)

			min_black = 1000000000




	for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):

		min_black=list_min_black[int((i)/20)]
		cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
		list_total=[]
		total_max=-1
		for (j, c) in enumerate(cnts):
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			mask = cv2.bitwise_and(thresh, thresh, mask=mask)
			total = cv2.countNonZero(mask)
			if total > total_max:
				total_max=total
			if total >0:
				list_total.append((total,j))

		answer_q = [char for char in ANSWER_KEY[q]]
		list_answer = []
		list_select=''
		for tt in list_total:
			if  tt[0] > min_black * 1.5 and  tt[0]>total_max*0.7:
				list_answer.append(tt[1])
				list_select=list_select+revert_translate[tt[1]]
		for answer in answer_q:
			color = (0, 0, 255)
			k = translate[answer]
			if k in list_answer:
				color = (0, 255, 0)
			cv2.drawContours(image, [cnts[k]], -1, color, 3)
		select.append(list_select)
	return select,image




def get_sbd(image_sbd, debug):
	image = image_sbd
	height, width, channels = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.adaptiveThreshold(gray, maxValue=255,
										   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
										   thresholdType=cv2.THRESH_BINARY_INV ,
										   blockSize=15,
										   C=8)
	#
	# cv2.imshow("cropped", thresh)
	# cv2.waitKey(0)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	questionCnts = []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		if  w >=width/13 and h >= height/13 and ar >= 0.7 and ar <= 1.3 and w < width/2 and h<=height/8 :
			questionCnts.append(c)

	questionCnts = contours.sort_contours(questionCnts,
										  method="top-to-bottom")[0]
	# cv2.drawContours(image, questionCnts, -1,  (0, 255, 0), 3)
	# cv2.imshow("cropped", image)
	# cv2.waitKey(0)

	if debug['on']:
		cv2.drawContours(image, questionCnts, -1, (0, 255, 0), 3)
		cv2.imwrite(debug['folder'] +'contours_sbd1.jpeg', image)

	if len(questionCnts)!=100:
		thresh = cv2.threshold(gray, 0, 255,
							   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		questionCnts = []
		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			if w >= width / 13 and h >= height / 13 and ar >= 0.7 and ar <= 1.3 and w < width / 2 and h <= height / 8:
				questionCnts.append(c)

		questionCnts = contours.sort_contours(questionCnts,
											  method="top-to-bottom")[0]
		cv2.drawContours(image, questionCnts, -1, (0, 255, 0), 3)
		#cv2.imshow("cropped", image)
		#cv2.waitKey(0)

		if debug['on']:
			cv2.imwrite(debug['folder'] +'contours_sbd.jpeg', image)

	sbd = []
	thresh = cv2.threshold(gray, 0, 255,
						   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	for i in range(0,10):
		list_questionCnts=[]
		for j1 in range(0,10):
			list_questionCnts.append(questionCnts[i+j1*10])
		cnts = contours.sort_contours(list_questionCnts,method="top-to-bottom")[0]
		bubbled = None
		min = 100000000
		total=0
		for (j, c) in enumerate(cnts):

			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			mask = cv2.bitwise_and(thresh, thresh, mask=mask)
			total = cv2.countNonZero(mask)
			# print(j)
			# cv2.imshow("cropped", mask)
			# cv2.waitKey(0)
			if total <= min:
				min = total
			if bubbled is None or total > bubbled[0]:
				bubbled = (total, j)

		if bubbled[0] < min * 1.4:
			bubbled = (total, -1)

		sbd.append(bubbled[1])

		if bubbled[1]!=-1:
			color = list(np.random.random(size=3) * 256)
			cv2.drawContours(image, [cnts[bubbled[1]]], -1, color, 3)
	return sbd[::-1],image



def get_mdt(image_mdt, debug):
	image = image_mdt
	height, width, channels = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	thresh = cv2.adaptiveThreshold(gray, maxValue=255,
										   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
										   thresholdType=cv2.THRESH_BINARY_INV ,
										   blockSize=15,
										   C=8)
	#
	# cv2.imshow("cropped", thresh)
	# cv2.waitKey(0)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	questionCnts = []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		if w >=width/10 and h >= height/13 and ar >= 0.7 and ar <= 1.3 and w < width/2 and h<=height/2 :
			questionCnts.append(c)

	questionCnts = contours.sort_contours(questionCnts,
										  method="top-to-bottom")[0]
	if debug['on']:
		cv2.drawContours(image, questionCnts, -1,  (0, 255, 0), 3)
		cv2.imwrite(debug['folder'] + 'contours01_mdt.jpeg', image)

	if len(questionCnts)!=60:
		thresh = cv2.threshold(gray, 0, 255,
							   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		questionCnts = []
		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			if w >= width / 10 and h >= height / 13 and ar >= 0.7 and ar <= 1.3 and w < width / 2 and h <= height / 2:
				questionCnts.append(c)

		questionCnts = contours.sort_contours(questionCnts,
											  method="top-to-bottom")[0]

		if debug['on']:
			cv2.drawContours(image, questionCnts, -1,  (0, 255, 0), 3)
			#cv2.imwrite(debug['folder'] + 'contours222_mdt.jpeg', img_contours)
			cv2.imwrite(debug['folder'] + 'contours02_mdt.jpeg', image)

	mdt = []
	thresh = cv2.threshold(gray, 0, 255,
						   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	for i in range(0,6):
		list_questionCnts=[]
		for j1 in range(0,10):
			list_questionCnts.append(questionCnts[i+j1*6])
		cnts = contours.sort_contours(list_questionCnts,method="top-to-bottom")[0]
		bubbled = None
		min = 100000000
		total=0

		for (j, c) in enumerate(cnts):
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			mask = cv2.bitwise_and(thresh, thresh, mask=mask)
			total = cv2.countNonZero(mask)
			if total <= min:
				min = total
			if bubbled is None or total > bubbled[0]:
				bubbled = (total, j)
		if bubbled[0] < min * 1.4:
			bubbled = (total, -1)

		mdt.append(bubbled[1])

		if bubbled[1]!=-1:
			color = list(np.random.random(size=3) * 256)
			cv2.drawContours(image, [cnts[bubbled[1]]], -1, color, 3)

	return mdt[::-1],image


def resolved(link, debug):
    ANSWER_KEY = ["A", "B", "C", "D","A","C", "D", "B", "A","C","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","A",
                      "A", "B", "C", "D","A","C", "D", "B", "A","C","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","A",
                      "A", "B", "C", "D","A","C", "D", "B", "A","C","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","D",
                      "A", "B", "C", "D","A","C", "D", "B", "A","C","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","A","A", "B", "C", "D","A"]

    img = cv2.imread(link)

    if debug['on']:
        cv2.imwrite(debug['folder'] +'input.png', img)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh =cv2.threshold(gray, 0, 255,
    # 					   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # output_image = link[:-5]+"_dentrang"+link[-5:]
    # cv2.imwrite(output_image, thresh)
    # cv2.waitKey(0)

    #if debug['on']:
    #    cv2.imwrite(debug['folder'] +'threshold_gray.jpeg', thresh)

    img_height, img_width, img_channels = img.shape
    max_weight=1807
    max_heigh=2555
    # crop_sbd=(951,254,1430,821)
    crop_sbd = (int(951 / max_weight* img_width), int(254 / max_heigh * img_height), int(1430 / max_weight* img_width), int(821 / max_heigh * img_height))
    # crop_mdt=(1418,254,1726,821)
    crop_mdt = (
    int(1418 / max_weight* img_width), int(254 / max_heigh * img_height), int(1726 / max_weight* img_width),
    int(821 / max_heigh * img_height))

    # crop_1_30=(41,833,480,2470)
    crop_1_30 = (
    int(41 / max_weight* img_width), int(833 / max_heigh * img_height), int(480 / max_weight* img_width),
    int(2470 / max_heigh * img_height))
    # crop_31_60 = (466, 833, 870, 2470)
    crop_31_60 = (
    int(466 / max_weight* img_width), int(833 / max_heigh * img_height), int(870 / max_weight* img_width),
    int(2470 / max_heigh * img_height))
    # crop_61_90 = (867, 833, 1292, 2470)
    crop_61_90 = (
    int(867 / max_weight* img_width), int(833 / max_heigh * img_height), int(1292 / max_weight* img_width),
    int(2470 / max_heigh * img_height))
    # crop_91_120 = (1270, 833, 1708, 2470)
    crop_91_120 = (
    int(1270 / max_weight* img_width), int(833 / max_heigh * img_height), int(1708 / max_weight* img_width),
    int(2470 / max_heigh * img_height))


    crop_img_sbd = img[crop_sbd[1]:crop_sbd[3], crop_sbd[0]:crop_sbd[2]]
    # cv2.imshow("cropped", crop_img_sbd)
    # cv2.waitKey(0)

    if debug['on']:
        cv2.imwrite(debug['folder'] +'crop_img_sbd.jpeg', crop_img_sbd)

    sbd, image_sbd = get_sbd(crop_img_sbd, debug)
    # print(sbd)

    crop_img_mdt = img[crop_mdt[1]:crop_mdt[3], crop_mdt[0]:crop_mdt[2]]
    # cv2.imshow("cropped", crop_img_mdt)
    # cv2.waitKey(0)

    if debug['on']:
        cv2.imwrite(debug['folder'] +'crop_img_mdt.jpeg', crop_img_mdt)

    mdt, image_mdt = get_mdt(crop_img_mdt, debug)
    # print(mdt)
    #
    crop_img_1_30 = img[crop_1_30[1]:crop_1_30[3], crop_1_30[0]:crop_1_30[2]]
    # # cv2.imshow("cropped", crop_img_1_30)
    # # cv2.waitKey(0)

    if debug['on']:
        cv2.imwrite(debug['folder'] +'crop_img_1_30.jpeg', crop_img_1_30)

    ans_1_30, image_1_30 = get_result_trac_nghiem(crop_img_1_30,ANSWER_KEY[0:30], debug)

    #
    crop_img_31_60 = img[crop_31_60[1]:crop_31_60[3], crop_31_60[0]:crop_31_60[2]]
    #
    # cv2.imshow("cropped", crop_img_31_60)
    # cv2.waitKey(0)

    if debug['on']:
        cv2.imwrite(debug['folder'] +'crop_img_31_60.jpeg', crop_img_31_60)

    ans_31_60, image_31_60 = get_result_trac_nghiem(crop_img_31_60, ANSWER_KEY[30:60], debug)
    # #
    # #
    crop_img_61_90 = img[crop_61_90[1]:crop_61_90[3], crop_61_90[0]:crop_61_90[2]]
    ans_61_90, image_61_90 = get_result_trac_nghiem(crop_img_61_90, ANSWER_KEY[60:90], debug)
    # # cv2.imshow("cropped", crop_img_61_90)
    # # cv2.waitKey(0)
    # #

    if debug['on']:
        cv2.imwrite(debug['folder'] +'crop_img_61_90.jpeg', crop_img_61_90)

    crop_img_91_120 = img[crop_91_120[1]:crop_91_120[3], crop_91_120[0]:crop_91_120[2]]
    if debug['on']:
        cv2.imwrite(debug['folder'] +'crop_img_91_120.jpeg', crop_img_91_120)

    ans_91_120, image_91_120 = get_result_trac_nghiem(crop_img_91_120, ANSWER_KEY[90:120], debug)
    # # cv2.imshow("cropped", crop_img_91_120)
    # # cv2.waitKey(0)
    # #


    all_answer_key = ans_1_30+ans_31_60+ans_61_90+ans_91_120
    # print(ans_1_30)
    # print(ans_31_60)
    # print(ans_61_90)
    # print(ans_91_120)
    print("len=" + str(len(all_answer_key)) + " - " + str(len(ans_1_30)) + " - " + str(len(ans_31_60)) + " - " + str(len(ans_61_90)) + " - " + str(len(ans_91_120)))
    print(all_answer_key)

    string_sbd=''.join(map(str,sbd))
    string_mdt=''.join(map(str,mdt))
    string_answer_list='_'.join(map(str,all_answer_key))

    #output_image = link[:-4]+"_result"+link[-4:]
    #cv2.imwrite(output_image, img)
    #cv2.waitKey(0)

    success = True
    if string_sbd.find("-") > 0 or string_mdt.find("-") > 0:
        success = False

    assert_res = assert_result(string_answer_list)
    return { 'success': success, 'student_code': string_sbd, 'permutation_exam_code': string_mdt, 'answers': string_answer_list ,'len': len(all_answer_key), 'assert': assert_res}

def assert_result(string_answer_list):
    result_list = string_answer_list.split('_')
    my_answers = ["A", "B", "B", "C", "D",
            "B", "A", "C", "D", "B",
            "A", "B", "", "C", "A",
            "D", "C", "A", "A", "A",
            "C", "C", "C", "B", "C",
            "D", "C", "D", "D", "D",
            "A", "A", "A", "A", "D",
            "D", "B", "B", "B", "B",
            "D", "A", "B", "C", "D",
            "A", "B", "C", "D", "B",
            "C", "B", "B", "A", "B",
            "C", "B", "B", "C", "D",
            "A", "A", "B", "B", "C",
            "D", "C", "B", "C", "D",
            "A", "A", "B", "B", "B",
            "B", "C", "C", "D", "D",
            "A", "B", "B", "C", "D",
            "A", "B", "B", "B", "C",
            "A", "B", "C", "D", "A",
            "", "", "", "", "",
            "A", "B", "C", "D", "B",
            "", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
    ];

    question_size = len(my_answers)
    result_size = len(result_list)
    correct_count = 0
    wa = dict()

    for i in range(0, question_size):
        if my_answers[i] == result_list[i]:
            correct_count += 1
        else:
            wa[i] =  my_answers[i] + " | " + result_list[i]

    return { 'result': str(correct_count) + '/' + str(question_size), 'wa': wa }


def current_milli_time():
    return round(time.time() * 1000)

def read_sheet_info_from_image(link):
    print("read link: " + link)
    try:
        parent_folder = './request_debug/' + str(current_milli_time()) + '/'
        print("debug folder: " + parent_folder)
        os.mkdir(parent_folder)
        return resolved(link, { 'on': True, 'folder': parent_folder })
    except Exception as err:
        print (str(err))
        traceback.print_tb(err.__traceback__)

    return {'success': False, 'message': 'Please try again !'}

	
