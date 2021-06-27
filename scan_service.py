from imutils import contours
import numpy as np
import imutils
import cv2
import os
import time
import traceback
import scan4 as scanner

def resolved(path_img):
	result = scanner.scan_exam(path_img)

	print (str(result))
	string_answer_list='_'.join(map(str, result['answers']))
	#output_image = link[:-4]+"_result"+link[-4:]
	#cv2.imwrite(output_image, img)
	#cv2.waitKey(0)

	success = True
	if result['student_code'].find("-") > 0 or result['exam_code'].find("-") > 0:
		success = False

	assert_res = assert_result(string_answer_list)
	return { 'success': success, 'student_code': result['student_code'], 'permutation_exam_code': result['exam_code'], 'answers': result['answers'] ,'len': len(result['answers']), 'assert': assert_res}

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
		return resolved(link)
	except Exception as err:
		print (str(err))
		traceback.print_tb(err.__traceback__)

	return {'success': False, 'message': 'Please try again !'}

	
