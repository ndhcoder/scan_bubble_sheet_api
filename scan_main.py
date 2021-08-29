import numpy as np
import imutils
from imutils import contours
import cv2
import logging
import os
import time
import traceback
import scan_utils as su

debug_mode = False
path_log_file = os.path.realpath(__file__) + '/../scan_main.log'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, filename=path_log_file)
logger.setLevel(logging.DEBUG)

def scan_exam(path_img):
    su.log_info(logger, 'scan path_img: ' + path_img)
    final_res = {
        'student_code': '-',
        'exam_code': '-',
        'answers': []
    }

    try:
        su.mkdir_base_folder()
        input_img = cv2.imread(path_img)
        img_height, img_width = input_img.shape[0:2]

        # find 4 corners points and make transform
        block_cnts_corners_right, input_img = get_block_cnts_col(0, input_img, 2, 1500, 4000, 0, img_width, 0, img_height, 'corner_right_', 120)

        img_height, img_width = input_img.shape[0:2]
        block_cnts_corners_left, input_img = get_block_cnts_col(1, input_img, 2, 1500, 4000, 0, img_width, 0, img_height, 'corner_left_', 120)
        su.export_img('input_after_detect_4_corner.jpg', input_img)

        input_img = get_input_transform(input_img, block_cnts_corners_left + block_cnts_corners_right, 0)
        input_img = cv2.resize(input_img, (2100, 2960))
        su.export_img('input_after_transform.jpg', input_img)

        #div ans block
        img_height, img_width = input_img.shape[0:2]
        block_cnts_col_right, _ = get_block_cnts_col(0, input_img, 9, 800, 2000, 10, img_width - 10, 10, img_height - 10, 'cnt_right_', 120)
        block_cnts_col_left, _ = get_block_cnts_col(1, input_img, 7, 800, 2000, 10, img_width - 10,  10, img_height - 10, 'cnt_left_', 120)

        block_cnts_col_right = contours.sort_contours(block_cnts_col_right,
                                                method="top-to-bottom")[0]

        block_cnts_col_left = contours.sort_contours(block_cnts_col_left,
                                                method="top-to-bottom")[0]

        exam_info = get_exam_info_img(input_img, block_cnts_col_right)
        input_img_ans = get_input_transform(input_img, [block_cnts_col_left[0], block_cnts_col_left[6],
                                                        block_cnts_col_right[2], block_cnts_col_right[8]])
        su.export_img('input_ans_transform.jpg', input_img_ans)
        all_ans = get_all_ans_points(block_cnts_col_left, block_cnts_col_right, input_img_ans)

        final_res = {
            'student_code': exam_info['student_code'],
            'exam_code': exam_info['exam_code'],
            'answers': all_ans
        }

        su.log_info(logger, 'scan path_img result: ' + path_img)
        su.log_info(logger, str(final_res))
    except Exception as err:
        su.log_error(logger, 'scan path_img failure: ' + path_img)
        traceback.print_tb(err.__traceback__)

    return final_res

    #block_cnts_rows, input_img = get_block_cnts_row(input_img)
    #block_cnts = block_cnts_cols + block_cnts_rows
    #block_cnts = contours.sort_contours(block_cnts,
    #                                   method="top-to-bottom")[0]
    #print("block found size = " + str(len(block_cnts)) + "!")

    #img_height, img_width = input_img.shape[0:2]
    #max_weight = 1807
    #max_height = 2555

def get_exam_info_img(input_img, block_cnts_col_4):
    su.debug_print("\n\n\n==== get_exam_info_img ==== ")
    img_height, img_width = input_img.shape[0:2]
    exam_info = {
        'student_code': '',
        'exam_code': ''
    }

    block_cnts_col_4 = contours.sort_contours(block_cnts_col_4,
                                            method="top-to-bottom")[0]

    (tr_x, tr_y, tr_w, tr_h) = cv2.boundingRect(block_cnts_col_4[0])
    (br_x, br_y, br_w, br_h) = cv2.boundingRect(block_cnts_col_4[1])
    offset = 250
    offset2 = 800

    input_img_left_code = input_img[(tr_y - 7):(br_y + br_w + 7), (tr_x - offset2 - 150):(tr_x - offset2)]
    su.export_img('input_img_left_code.jpg', input_img_left_code)
    #img_height, img_width = input_img_left_code.shape[0:2]
    block_cnts_code_left, _ = get_block_cnts_col(3, input_img, 2, 950, 4000,
                                                 (tr_x - offset2 - 150), (tr_x - offset2), (tr_y - 7), (br_y + br_h + 7), 'code_left_', 60)

    su.export_img_cnt('block_cnts_code_exam.jpg', input_img, block_cnts_code_left, True)

    input_exam_img = get_input_transform(input_img, block_cnts_code_left + block_cnts_col_4[0:2], 7)
    su.export_img('input_exam_img.jpg', input_exam_img)

    img_height, img_width = input_exam_img.shape[0:2]
    div = img_width // 19
    exam_code_img = input_exam_img[0:img_height, (div * 12 - div // 2) : img_width]
    student_code_img = input_exam_img[0:img_height, 0 : (div * 12 - div // 2)]
    su.export_img('exam_code_img_before_cut.jpg', exam_code_img)
    su.export_img('student_code_img_before_cut.jpg', student_code_img)

    exam_code_img = su.get_bigest_frame('exam_code', exam_code_img)
    student_code_img = su.get_bigest_frame('student_code_img', student_code_img)

    su.export_img('exam_code_img_after_cut.jpg', exam_code_img)
    su.export_img('student_code_img_after_cut.jpg', student_code_img)

    student_code = get_sbd_detail('student_code', student_code_img)
    exam_code = get_exam_code_detail('exam_code', exam_code_img)

    exam_info['student_code'] = student_code
    exam_info['exam_code'] = exam_code

    return exam_info

def get_exam_code_detail(name, exam_code_block_img):
    ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(exam_code_block_img)
    su.export_img(name + '_block_bg_img.png', ans_block_bg_img)

    thresh_bg = cv2.adaptiveThreshold(ans_block_bg_img, maxValue=255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=15,
                                   C=8)

    su.export_img(name + '_thresh_bg.png', thresh_bg)
    img_height, img_width, img_channels = exam_code_block_img.shape
    size_1_col = int(img_width / 6)
    size_1_row = int(img_height / 10)
    exam_code = list("------")
    exam_code_percent = [0,0,0,0,0,0]

    padding = 8
    for i in range(0, 10):
        for j in range(0, 6):
            square = thresh_bg[int((i * size_1_row)):int(((i + 1) * size_1_row)), int((j * size_1_col)):int(((j + 1) * size_1_col))]
            square_height, square_width = square.shape
            square = square[padding:(square_height-padding), padding:(square_width-padding)]
            su.export_img(name + '_square_' + str(i) + '_' + str(j) + '.jpg', square)
            percent_non_zero = su.get_percent_non_zero(square)
            if percent_non_zero > exam_code_percent[j]:
                exam_code_percent[j] = percent_non_zero
                exam_code[j] = str(i)

    return ''.join(exam_code)

def get_sbd_detail(name, sbd_block_img):
    ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(sbd_block_img)
    su.export_img(name + '_block_bg_img.png', ans_block_bg_img)

    thresh_bg = cv2.adaptiveThreshold(ans_block_bg_img, maxValue=255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=15,
                                   C=8)

    su.export_img(name + '_thresh_bg.png', thresh_bg)
    img_height, img_width, img_channels = sbd_block_img.shape
    size_1_col = int(img_width / 10)
    size_1_row = int(img_height / 10)
    sbd = list("----------")
    sbd_percent = [0,0,0,0,0,0,0,0,0,0]
    padding = 8
    for i in range(0, 10):
        for j in range(0, 10):
            square = thresh_bg[int((i * size_1_row)):int(((i + 1) * size_1_row)), int((j * size_1_col)):int(((j + 1) * size_1_col))]
            square_height, square_width = square.shape
            square = square[padding:(square_height-padding), padding:(square_width-padding)]
            su.export_img(name + '_square_' + str(i) + '_' + str(j) + '.jpg', square)
            percent_non_zero = su.get_percent_non_zero(square)
            if percent_non_zero > sbd_percent[j]:
                sbd_percent[j] = percent_non_zero
                sbd[j] = str(i)

    return ''.join(sbd)


def get_all_ans_points(block_cnts_col_0, block_cnts_col_4, input_ans_img):
    su.debug_print("\n\n\n==== get_all_ans_points ==== ")
    img_height, img_width = input_ans_img.shape[0:2]
    offset_4 = img_width // 4
    all_ans = []

    for i in range(1, 5):
        img_temp = input_ans_img[0:img_height, ((i - 1) * offset_4):(i * offset_4)]
        su.export_img('ans_col_' + str(i) + '.jpg', img_temp)
        ans_block_col = su.get_bigest_frame('ans_col_biggest_frame_' + str(i), img_temp)

        #remove index question
        _h, a_w = ans_block_col.shape[0:2]
        _sw = a_w // 5
        ans_block_col = ans_block_col[0:_h, _sw:a_w]

        su.export_img('ans_col_biggest_frame_' + str(i) + '.jpg', ans_block_col)
        ans_points, cnts_correct = get_ans('col_' + str(i) + '_ans_block', ans_block_col, i)

        su.debug_print('col_' + str(i) + ' ans: '  + str(ans_points))
        all_ans.extend(ans_points)

    return all_ans

def get_ans(name, ans_block_img, col_index):
    # su.export_img(name + '.png', input_img)
    ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(ans_block_img)

    #su.export_img(name + 'ans_block_binary_img.png', ans_block_binary_img)
    #su.export_img(name + 'ans_block_outgray_img.png', ans_block_outgray_img)
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
    cnts = contours.sort_contours(cnts,
                                  method="top-to-bottom")[0]
    su.export_img_cnt(name + '_cnts.png', ans_block_img, cnts, True)

    question_cnts = binary_search_question_cnt(cnts)
    su.export_img_cnt(name + '_question_cnts.png', ans_block_img, question_cnts, True)
    #
    # if col_index == 5:
    #     map_xy = []
    #
    #     for cnt in question_cnts:
    #         (x, y, w, h) = cv2.boundingRect(cnt)
    #         map_xy.append({'x': x, 'y': y})
    #
    #     for cnt in question_cnts:
    #         (x, y, w, h) = cv2.boundingRect(cnt)
    #         # print ("cnt: " + str(min_dist_debug(x,y, map_xy)))

    size_question_cnts = len(question_cnts)
    su.debug_print('size_question_cnts: ' + str(size_question_cnts))

    if size_question_cnts != 120:
        raise Exception('find size_question_cnts error: ' + str(size_question_cnts))

    index_question = 0
    question_cnts = contours.sort_contours(question_cnts,
                                           method="top-to-bottom")[0]
    results = []
    percents = []
    only_percents = []
    max_percent = 0
    result_item_text = ['A', 'B', 'C', 'D']

    min_size = find_min_area(question_cnts)
    for (q, i) in enumerate(np.arange(0, size_question_cnts, 4)):
        question_row_cnts = contours.sort_contours(question_cnts[i:(i + 4)],
                                                   method="left-to-right")[0]

        padding_question = 7
        for cnt in question_row_cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if (y + padding_question) < (y + h - padding_question) and (x + padding_question) < (x + w - padding_question):
                question_item = thresh_bg[(y + padding_question):(y + h - padding_question), (x + padding_question):(x + w - padding_question)]
            else:
                question_item = thresh_bg[(y):(y + h), (x):(x + w)]

            su.export_img(name + '_question' + str(index_question + 1) + '.png', question_item)

            max_area, max_w, max_h = find_max_cnt_area(question_item)
            su.debug_print(name + '_question' + str(index_question + 1) + " ar=" + str(max_w / float(max_h)))
            percent_non_zero = su.get_percent_non_zero_with_size(question_item, min_size)
            # non_zero = su.get_percent_non_zero_mask(str(index_question), question_item, ans_block_img[y:(y + h), x:(x + w)])
            # su.debug_print(str(index_question + 1) + " with percent: " + str(percent_non_zero))
            index_question += 1
            percents.append({'value': percent_non_zero, 'cnt': cnt})
            only_percents.append(percent_non_zero)
            if percent_non_zero > max_percent:
                max_percent = percent_non_zero

    max_percent = 27 if max_percent < 27 else max_percent
    temp = max_percent / 1.8
    point_avg = find_mid_points(only_percents, 15, 5)
    su.debug_print("point_avg: " + str(point_avg))

    if point_avg > 0:
        min_percent_correct = temp if point_avg < temp else point_avg
        # if min_percent_correct < cache_percent['min']:
        # 	min_percent_correct = cache_percent['min']
        # else:
        # 	cache_percent['min'] = min_percent_correct

        # if max_percent > cache_percent['max']:
        # 	cache_percent['max'] = max_percent
        # else:
        # 	max_percent = cache_percent['max']
        # 	cache_percent['max'] = max_percent
    else:
        min_percent_correct = -1

    cnts_correct = []
    su.debug_print("min_percent_correct: " + str(min_percent_correct) + ", max_percent=" + str(max_percent))
    for (q, i) in enumerate(np.arange(0, size_question_cnts, 4)):
        index_ans = 0
        result_item = ''
        temp_percents = su.sub_list(percents, i, i + 4)
        for percent in temp_percents:
            su.debug_print("q: " + str(len(results) + 1) + ": " + str(percent['value']))
            if percent['value'] > min_percent_correct and min_percent_correct > 0:
                result_item += result_item_text[index_ans]
                cnts_correct.append(percent['cnt'])
            index_ans += 1

        results.append(result_item)

    return results, cnts_correct

def find_mid_points(arr, dis_avg, dis_arr):
    result = -1
    max_dis = 0
    arr.sort()
    for a in arr:
        cal = cal_mid_points(arr, a, dis_avg, dis_arr)

        if cal['accept'] and cal['distance_arr'] > max_dis:
            result = a
            max_dis = cal['distance_arr']

    return result

def cal_mid_points(arr, mid, dis_avg, dis_arr):
    arr1 = []
    arr2 = []
    count1 = 0
    count2 = 0
    sum1 = 0
    sum2 = 0
    max_arr1 = 0
    min_arr2 = 999999999

    for a in arr:
        if a <= mid:
            arr1.append(a)
            count1 += 1
            sum1 += a
            if a > max_arr1:
                max_arr1 = a
        else:
            count2 += 1
            sum2 += a
            arr2.append(a)
            if a < min_arr2:
                min_arr2 = a

    if sum1 == 0 or sum2 == 0:
        return {'accept': False}

    avg1 = sum1 / count1
    avg2 = sum2 / count2

    real_dis_avg = avg2 - avg1
    real_dis_arr = min_arr2 - max_arr1

    su.debug_print ("find_cal_point: mid=" + str(mid) + " dis_avg=" + str(real_dis_avg) + ", dis_arr=" + str(real_dis_arr) + ", max_arr1=" + str(max_arr1) + ", min_arr2=" + str(min_arr2))

    if avg2 - avg1 < dis_avg:
        return {'accept': False}

    if min_arr2 - max_arr1 < dis_arr:
        return {'accept': False}

    return {'accept': True, 'distance_avg': avg2 - avg1, 'distance_arr': min_arr2 - max_arr1}

def find_max_cnt_area(thresh_img):
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_area = 0
    max_w = 0
    max_h = 1
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if (w * h > max_area):
            max_area = w * h
            max_w = w
            max_h = h

    return max_area, max_w, max_h

def binary_search_question_cnt(cnts):
    left = 0
    right = 5000
    mid = (left + right ) // 2
    question_cnts = []
    while left < right and left != mid and mid != right:
        question_cnts = find_question_cnt(cnts, mid)
        size_question_cnts = len(question_cnts)

        #print ("mid = " + str(mid) + ", size = " + str(size_question_cnts))
        if size_question_cnts == 120:
            return question_cnts

        if size_question_cnts < 120:
            right = mid
            mid = (left + right) // 2
        else:
            left = mid
            mid = (left + right) // 2

    return question_cnts

def find_question_cnt(cnts, min_area):
    question_cnts = []
    map_xy = []
    index = 0
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ar = w / float(h)
        area = w * h #cv2.contourArea(cnt)
        mind = min_dist(x, y, map_xy)
        if (x > 0 or y > 0) and ar > 0.5 and ar < 1.5 and area > min_area and (index == 0 or mind > 500):
            #print (str(area) + ", w=" + str(w) + ", h=" + str(h))
            map_xy.append({'x': x, 'y': y})
            question_cnts.append(cnt)
            index+=1

    return question_cnts

def find_min_area(cnts):
    min_size = 9999999999999
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if w * h < min_size:
            min_size = w * h

    return min_size

def get_input_transform(input_img, block_cnts, offset_bl = 0):
    pts = get_4_corners_points(input_img, block_cnts, offset_bl)
    warped = su.four_point_transform(input_img, pts)
    h,w,c = warped.shape
    #print ('shape: ' + str(warped.shape))
    return warped[0:h, 0:w]

def get_4_corners_points(input_img, block_cnts, offset_bl):
    block_cnts = contours.sort_contours(block_cnts,
                                            method="left-to-right")[0]
    block_cnts_left = block_cnts[0:2]
    block_cnts_right = block_cnts[2:4]

    block_cnts_left_sorted = contours.sort_contours(block_cnts_left,
                                            method="top-to-bottom")[0]
    block_cnts_right_sorted = contours.sort_contours(block_cnts_right,
                                            method="top-to-bottom")[0]

    tl_x, tl_y, tl_w, tl_h = cv2.boundingRect(block_cnts_left_sorted[0])
    bl_x, bl_y, bl_w, bl_h = cv2.boundingRect(block_cnts_left_sorted[1])
    bl_x = bl_x + offset_bl

    tr_x, tr_y, tr_w, tr_h = cv2.boundingRect(block_cnts_right_sorted[0])
    br_x, br_y, br_w, br_h = cv2.boundingRect(block_cnts_right_sorted[1])

    penaty_left = 0
    penaty_right = 0
    penaty_vertical = 0
    top_left = (tl_x - penaty_left, tl_y - penaty_vertical)
    top_right = (tr_x + tr_w + penaty_right, tr_y - penaty_vertical)
    bottom_right = (br_x + br_w + penaty_right, br_y + br_h + penaty_vertical)
    bottom_left = (bl_x - penaty_left, bl_y + bl_h + penaty_vertical)

    su.export_img('corners_input_before.png',
                  input_img[top_left[1]:bottom_left[1], top_left[0]:top_right[0]])
    return su.order_points(np.array([top_left, top_right, bottom_right, bottom_left]))

def get_block_cnts_col(direction, input_img, expectedSize, minAreaLimit, maxArea, minX, maxX, minY, maxY, debugName, offset):
    img_height, img_width = input_img.shape[0:2]
    block_cnts_cols = []
    size_block = 0

    img_temp = input_img
    maxX_temp = maxX
    minX_temp = minX
    for i in range(0, 20):
        if direction == 0: #right
            img_temp = input_img[0:img_height, 0:(img_width - i * 10)]
        elif direction == 1:
            img_temp = input_img[0:img_height, (i * 10):img_width]
        elif direction == 2:
            maxX_temp = maxX - i * 10
        else:
            minX_temp = minX + i * 10

        block_cnts_cols = get_block_cnts_col_main(direction, img_temp, expectedSize, minAreaLimit, maxArea,
                                                  minX_temp, maxX_temp, minY, maxY, debugName, offset)
        size_block = len(block_cnts_cols)

        if size_block == expectedSize:
            break

    if size_block != expectedSize:
        raise Exception('find size block col error: ' + str(size_block))

    return block_cnts_cols, img_temp

def get_block_cnts_col_main(direction, input_img, expectedSize, minAreaLimit, maxArea, minX, maxX, minY, maxY, debugName, offset = 120):
    img_height, img_width = input_img.shape[0:2]
    ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(input_img)
    thresh = cv2.adaptiveThreshold(ans_block_binary_img, maxValue=255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=15,
                                   C=8)

    su.export_img(debugName + 'input_img_thresh_col' + str(direction) + '.png', thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    border_limit = maxX - offset if direction == 0 or direction == 2 else minX + offset
    block_cnts = binary_search_block_cnt_col(direction, img_width, img_height, border_limit, cnts, expectedSize,
                                             minAreaLimit, maxArea, minX, maxX, minY, maxY)
    size_block = len(block_cnts)
    su.debug_print(debugName + 'size_block col: ' + str(size_block))
    test = su.export_img_cnt(debugName + 'block_cnts_col' + str(direction) + '.png', input_img, block_cnts, True)

    if size_block != expectedSize:
        return block_cnts

    block_cnts = contours.sort_contours(block_cnts,
                                        method="top-to-bottom")[0]

    su.export_img_cnt(debugName + 'block_cnt2_col' + str(direction) + '.png', input_img, block_cnts, True)
    return block_cnts

def binary_search_block_cnt_col(direction, img_width, img_height, border_limit, cnts, expectedSize, minAreaLimit, maxArea, minX, maxX, minY, maxY):
    left = minAreaLimit
    right = maxArea
    mid = (left + right ) // 2
    result_cnts = []
    while left < right and left != mid and mid != right and mid >= minAreaLimit:
        result_cnts = find_block_cnt_col(direction, img_width, img_height, border_limit, cnts, mid, maxArea, minX, maxX, minY, maxY)
        size_cnts = len(result_cnts)

        su.debug_print("mid = " + str(mid) + ", size = " + str(size_cnts))
        if size_cnts == expectedSize:
            su.debug_print("mid = " + str(mid) + ", size = " + str(size_cnts))
            assert_size = assertSize(result_cnts)
            su.debug_print(str(assert_size))
            return result_cnts

        if size_cnts < expectedSize:
            right = mid
            mid = (left + right) // 2
        else:
            left = mid
            mid = (left + right) // 2

    return result_cnts

def assertSize(cnts):
    res = []
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        size = w * h
        res.append(size)

    return res

def find_block_cnt_col(direction, img_width, img_height, border_limit, cnts, min_area, maxArea, minX, maxX, minY, maxY):
    block_cnts = []
    map_xy = []
    index = 0

    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ar = w / float(h)
        area = w * h #cv2.contourArea(cnt)
        mind = min_dist(x, y, map_xy)

        if direction == 0:
            if (index == 0 or mind > 500) and y >= minY and y <= maxY and ar > 0.5 and ar < 4 and x >= minX and x <= maxX and area <= maxArea and area >= min_area and (x >= border_limit) and (x + w) < (maxX) and (y + h) < (img_height):
                #print (str(area) + ", w=" + str(w) + ", h=" + str(h))
                block_cnts.append(cnt)
                #print ("mind: " + str(mind))
                map_xy.append({'x': x, 'y': y})
                index += 1
        else:
            if (index == 0 or mind > 500) and y >= minY and y <= maxY and ar > 0.5 and ar < 4 and x >= minX and x <= maxX and area <= maxArea and area >= min_area and (x + w) <= border_limit and (
                y + h) < (img_height):
                # print (str(area) + ", w=" + str(w) + ", h=" + str(h))
                block_cnts.append(cnt)
                # print ("mind: " + str(mind))
                map_xy.append({'x': x, 'y': y})
                index += 1

    return block_cnts

def min_dist(x, y, map_xy):
    m = 99999999
    for a in map_xy:
        d = (a['x'] - x) * (a['x'] - x) + (a['y'] - y) * (a['y'] - y)
        if d < m:
            m = d

    return m

def min_dist_debug(x, y, map_xy):
    m = 99999999
    for a in map_xy:
        if a['x'] == x and a['y'] == y:
            continue

        d = (a['x'] - x) * (a['x'] - x) + (a['y'] - y) * (a['y'] - y)
        if d < m:
            m = d

    return m

#path_img = 'E:\\hgedu-6\\0001.jpg'
#scan_exam(path_img)
