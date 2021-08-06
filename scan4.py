import numpy as np
import imutils
from imutils import contours
import cv2
import os
import time
import traceback
import scan_utils as su

debug_mode = False

def scan_exam(path_img):
    su.mkdir_base_folder()
    input_img = cv2.imread(path_img)
    block_cnts_cols, input_img = get_block_cnts_col(input_img)
    block_cnts_rows, input_img = get_block_cnts_row(input_img)
    block_cnts = block_cnts_cols + block_cnts_rows
    block_cnts = contours.sort_contours(block_cnts,
                                        method="top-to-bottom")[0]
    print("block found size = " + str(len(block_cnts)) + "!")

    #input_img = su.convert_to_binary_img(input_img)
    img_height, img_width = input_img.shape[0:2]
    max_weight=1807
    max_heigh=2555

    #get sbd
    crop_sbd_position = (int(951 / max_weight* img_width), int(254 / max_heigh * img_height), int(1430 / max_weight* img_width), int(821 / max_heigh * img_height))
    crop_sbd_img = input_img[crop_sbd_position[1]:crop_sbd_position[3], crop_sbd_position[0]:crop_sbd_position[2]]
    su.export_img('crop_sbd_img.png', crop_sbd_img)
    sbd = get_sbd(crop_sbd_img)
    print ("SBD: " + sbd)

    #get exam code
    crop_exam_code = (
        int(1418 / max_weight* img_width), int(254 / max_heigh * img_height), int(1726 / max_weight* img_width),
        int(821 / max_heigh * img_height))
    crop_img_exam_code = input_img[crop_exam_code[1]:crop_exam_code[3], crop_exam_code[0]:crop_exam_code[2]]
    su.export_img('crop_exam_code_img.png', crop_img_exam_code)
    exam_code = get_exam_code(crop_img_exam_code)
    print ("Exam Code: " + exam_code)

    offset_bottoms = [300, 500, 600, 650, 700, 750, 800]
    size_block = 0
    #get answers

    # for offset_bottom in offset_bottoms:
    #     block_cnts = binary_search_get_block_cnts(input_img, offset_bottom)  # get_block_cnts(input_img, 200)
    #     size_block = len(block_cnts)
    #     print ("try get cnts with offset: " + str(offset_bottom) + " size=" + str(size_block))
    #     if size_block == 59:
    #         break
    #
    # if size_block != 59:
    #     raise Exception('find size block error: ' + str(size_block))
    #
    ans_block_names = ['', '1_30', '31_60', '61_90', '91_120']
    all_ans = []
    cache_percent = {'min': 0, 'max': 0}

    for col_index in range(1, 5):
        name_block = ans_block_names[col_index]
        ans_block_img = get_ans_block(input_img, block_cnts, col_index)
        su.export_img(name_block + '_ans_block_img.png', ans_block_img)

        ans, cnts_correct = get_ans(name_block, ans_block_img, col_index, cache_percent)
        su.export_img_cnt('000_' + name_block + '_final.png', ans_block_img, cnts_correct, False)
        all_ans.extend(ans)

    final_res = {
        'answers': all_ans,
        'student_code': sbd,
        'exam_code': exam_code
    }

    su.debug_print (str(final_res))
    return final_res


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

def assertSize(cnts):
    res = []
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        size = w * h
        res.append(size)

    return res


def binary_search_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts):
    left = 300
    right = 5000
    mid = (left + right ) // 2
    expect_size = 59
    result_cnts = []
    while left < right and left != mid and mid != right:
        result_cnts = find_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts, mid)
        size_cnts = len(result_cnts)

        #su.debug_print ("mid = " + str(mid) + ", size = " + str(size_cnts))
        if size_cnts == expect_size:
            su.debug_print ("mid = " + str(mid) + ", size = " + str(size_cnts))
            assertSize(result_cnts);
            return result_cnts

        if size_cnts < expect_size:
            right = mid
            mid = (left + right) // 2
        else:
            left = mid
            mid = (left + right) // 2

    return result_cnts


def binary_search_block_cnt_col(img_width, img_height, right_border_from, cnts):
    left = 300
    right = 5000
    mid = (left + right ) // 2
    expect_size = 42
    result_cnts = []
    while left < right and left != mid and mid != right:
        result_cnts = find_block_cnt_col(img_width, img_height, right_border_from, cnts, mid)
        size_cnts = len(result_cnts)

        #su.debug_print ("mid = " + str(mid) + ", size = " + str(size_cnts))
        if size_cnts == expect_size:
            su.debug_print ("mid = " + str(mid) + ", size = " + str(size_cnts))
            assertSize(result_cnts);
            return result_cnts

        if size_cnts < expect_size:
            right = mid
            mid = (left + right) // 2
        else:
            left = mid
            mid = (left + right) // 2

    return result_cnts

def binary_search_block_cnt_row(img_width, img_height, bottom_border_from, cnts):
    left = 300
    right = 5000
    mid = (left + right ) // 2
    expect_size = 18
    result_cnts = []
    while left < right and left != mid and mid != right:
        result_cnts = find_block_cnt_row(img_width, img_height, bottom_border_from, cnts, mid)
        size_cnts = len(result_cnts)

        #su.debug_print ("mid = " + str(mid) + ", size = " + str(size_cnts))
        if size_cnts == expect_size:
            su.debug_print ("mid = " + str(mid) + ", size = " + str(size_cnts))
            assertSize(result_cnts);
            return result_cnts

        if size_cnts < expect_size:
            right = mid
            mid = (left + right) // 2
        else:
            left = mid
            mid = (left + right) // 2

    return result_cnts

def find_block_cnt_col(img_width, img_height, right_border_from, cnts, min_area):
    block_cnts = []
    map_xy = []
    index = 0
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ar = w / float(h)
        area = w * h #cv2.contourArea(cnt)
        mind = min_dist(x, y, map_xy)
        if (index == 0 or mind > 500) and area >= min_area and (x >= right_border_from) and ar > 1.3 and ar < 4 and (x + w) < (img_width) and (y + h) < (img_height):
            #print (str(area) + ", w=" + str(w) + ", h=" + str(h))
            block_cnts.append(cnt)
            #print ("mind: " + str(mind))
            map_xy.append({'x': x, 'y': y})
            index += 1

    return block_cnts

def find_block_cnt_row(img_width, img_height, bottom_border_from, cnts, min_area):
    block_cnts = []
    map_xy = []
    index = 0
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ar = w / float(h)
        area = w * h #cv2.contourArea(cnt)
        mind = min_dist(x, y, map_xy)
        if (index == 0 or mind > 500) and area >= min_area and (y >= bottom_border_from) and ar > 1.3 and ar < 4 and (x + w) < (img_width) and (y + h) < (img_height):
            #print (str(area) + ", w=" + str(w) + ", h=" + str(h))
            block_cnts.append(cnt)
            #print ("mind: " + str(mind))
            map_xy.append({'x': x, 'y': y})
            index += 1

    return block_cnts

def find_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts, min_area):
    block_cnts = []
    map_xy = []
    index = 0
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ar = w / float(h)
        area = w * h #cv2.contourArea(cnt)
        mind = min_dist(x, y, map_xy)
        if (index == 0 or mind > 500) and area >= min_area and (x > right_border_from or y > bottom_border_from) and ar > 1.3 and ar < 4 and (x + w) < (img_width) and (y + h) < (img_height):
            #print (str(area) + ", w=" + str(w) + ", h=" + str(h))
            block_cnts.append(cnt)
            #print ("mind: " + str(mind))
            map_xy.append({'x': x, 'y': y})
            index += 1

    return block_cnts

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

def get_ans(name, ans_block_img, col_index, cache_percent):
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
    cnts = contours.sort_contours(cnts,
        method="top-to-bottom")[0]
    su.export_img_cnt(name + '_cnts.png', ans_block_img, cnts, True)

    question_cnts = binary_search_question_cnt(cnts)
    su.export_img_cnt(name + '_question_cnts.png', ans_block_img, question_cnts, True)

    if col_index == 5:
        map_xy = []

        for cnt in question_cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            map_xy.append({'x': x, 'y': y})

        for cnt in question_cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            #print ("cnt: " + str(min_dist_debug(x,y, map_xy)))

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
        question_row_cnts = contours.sort_contours(question_cnts[i:(i+4)],
                                              method="left-to-right")[0]

        for cnt in question_row_cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            question_item = thresh_bg[y:(y + h), x:(x + w)]
            #su.export_img(name + '_question' + str(index_question + 1) + '.png', question_item)
            max_area, max_w, max_h = find_max_cnt_area(question_item)
            su.debug_print(name + '_question' + str(index_question + 1) + " ar=" + str(max_w / float(max_h)))
            percent_non_zero = su.get_percent_non_zero_with_size(question_item, min_size)
            #non_zero = su.get_percent_non_zero_mask(str(index_question), question_item, ans_block_img[y:(y + h), x:(x + w)])
            #su.debug_print(str(index_question + 1) + " with percent: " + str(percent_non_zero))
            index_question += 1
            percents.append({'value': percent_non_zero, 'cnt': cnt})
            only_percents.append(percent_non_zero)
            if percent_non_zero > max_percent:
                max_percent = percent_non_zero

    max_percent = 27 if max_percent < 27 else max_percent
    temp = max_percent / 1.8
    min_percent_correct = temp if temp > 20 else 20



    point_avg = find_mid_points(only_percents, 15, 2)
    print ("point_avg: " + str(point_avg))
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
    print ("min_percent_correct: " + str(min_percent_correct) + ", max_percent=" + str(max_percent))
    for (q, i) in enumerate(np.arange(0, size_question_cnts, 4)):
        index_ans = 0
        result_item = ''
        temp_percents = su.sub_list(percents, i, i + 4)
        for percent in temp_percents:
            su.debug_print ("q: " + str(len(results) + 1) + ": " + str(percent['value']))
            if percent['value'] > min_percent_correct and min_percent_correct > 0:
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

def binary_search_get_block_cnts(input_img, right):
    left = 50
    mid = (left + right ) // 2
    expect_size = 59
    result_cnts = []
    while left < right and left != mid and mid != right:
        result_cnts = get_block_cnts(input_img, mid)
        size_cnts = len(result_cnts)

        #print ("mid = " + str(mid) + ", size = " + str(size_cnts))
        if size_cnts == expect_size:
            return result_cnts

        if size_cnts < expect_size:
            right = mid
            mid = (left + right) // 2
        else:
            left = mid
            mid = (left + right) // 2

    return result_cnts

def get_block_cnts_col(input_img):
    img_height, img_width = input_img.shape[0:2]
    block_cnts_cols = []
    size_block = 0
    img_temp = input_img
    for i in range(0, 20):
        img_temp = input_img[0:img_height, 0:(img_width - i * 10)]
        block_cnts_cols = get_block_cnts_col_main(img_temp)
        size_block = len(block_cnts_cols)

        if size_block == 42:
            break

    if size_block != 42:
        raise Exception('find size block col error: ' + str(size_block))
    return block_cnts_cols, img_temp

def get_block_cnts_row(input_img):
    img_height, img_width = input_img.shape[0:2]
    block_cnts_rows = []
    size_block = 0
    img_temp = input_img
    for i in range(0, 30):
        img_temp = input_img[0:(img_height- i * 10), 0:img_width]
        block_cnts_rows = get_block_cnts_row_main(img_temp)
        size_block = len(block_cnts_rows)

        if size_block == 18:
            break

    if size_block != 18:
        raise Exception('find size block row error: ' + str(size_block))

    return block_cnts_rows[0:17], img_temp

def get_block_cnts_col_main(input_img):
    img_height, img_width = input_img.shape[0:2]
    ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(input_img)
    thresh = cv2.adaptiveThreshold(ans_block_binary_img, maxValue=255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=15,
                                   C=8)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    img_right = input_img[0:img_height, (img_width - 120):img_width]
    block_cnts = binary_search_block_cnt_col(img_width, img_height, img_width - 120, cnts)
    size_block = len(block_cnts)
    su.debug_print('size_block col: ' + str(size_block))
    test = su.export_img_cnt('block_cnts_col.png', input_img, block_cnts, True)
    test2 = su.export_img('block_cnts_crop_col.png', input_img[0])

    if size_block != 42:
        return block_cnts #raise Exception('find size block error: ' + str(size_block))

    block_cnts = contours.sort_contours(block_cnts,
                                        method="top-to-bottom")[0]
    su.export_img_cnt('block_cnt2_col.png', input_img, block_cnts, True)
    return block_cnts
#
def get_block_cnts_row_main(input_img):
    img_height, img_width = input_img.shape[0:2]
    ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(input_img)
    thresh = cv2.adaptiveThreshold(ans_block_binary_img, maxValue=255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=15,
                                   C=8)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    img_bottom = input_img[(img_height - 150):img_height, 0:img_width]
    block_cnts = binary_search_block_cnt_row(img_width, img_height, img_height - 150, cnts)
    size_block = len(block_cnts)
    su.debug_print('size_block row: ' + str(size_block))
    test = su.export_img_cnt('block_cnts_row.png', input_img, block_cnts, True)
    test2 = su.export_img('block_cnts_crop_row.png', input_img[0])

    if size_block != 18:
        return block_cnts #raise Exception('find size block error: ' + str(size_block))

    block_cnts = contours.sort_contours(block_cnts,
                                        method="left-to-right")[0]
    su.export_img_cnt('block_cnt2_row.png', input_img, block_cnts, True)
    return block_cnts
#
# Get 59 border around block points
#
def get_block_cnts(input_img, offset_bottom):
    img_height, img_width = input_img.shape[0:2]
    right_border_from = int(img_width - img_width/15)
    bottom_border_from = img_height - offset_bottom

    img_height_max = bottom_border_from + 100 if bottom_border_from + 100 < img_height else img_height
    crop_border_right = input_img[0:img_height_max, right_border_from:(img_width)]
    crop_border_bottom = input_img[bottom_border_from:(img_height_max), 0:img_width]
    su.export_img('crop_border_right.png', crop_border_right)
    su.export_img('crop_border_bottom.png', crop_border_bottom)

    ans_block_binary_img, ans_block_outgray_img, ans_block_bg_img = su.convert_to_binary_img(input_img)
    thresh = cv2.adaptiveThreshold(ans_block_binary_img, maxValue=255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=15,
                                   C=8)

    #su.export_img('block_edged_img.png', edged_img)
    su.export_img('block_thresh.png', thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    su.export_img_cnt('block_cnts_all.png', input_img, cnts, True)
    block_cnts = binary_search_block_cnt(img_width, img_height, right_border_from, bottom_border_from, cnts)
    size_block = len(block_cnts)
    su.debug_print('size_block: ' + str(size_block))
    su.export_img_cnt('block_cnts.png', input_img, block_cnts, True)
    su.export_img('block_cnts_crop.png', input_img[0])

    if size_block != 59:
        return block_cnts#$Exception('find size block error: ' + str(size_block))

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

    penaty_left = 20
    penaty_right = 40
    penaty_vertical = 40
    top_left = (first_row_x - penaty_left, first_col_y - penaty_vertical)
    top_right = (last_row_x + last_row_w + penaty_right, first_col_y - penaty_vertical)
    bottom_right = (last_row_x + last_row_w + penaty_right, last_row_y + penaty_vertical)
    bottom_left = (first_row_x - penaty_left, first_row_y + penaty_vertical)

    su.export_img(str(col_index) + '_block_before.png', input_img[top_left[1]:bottom_left[1], top_left[0]:top_right[0]])
    return su.order_points(np.array([top_left, top_right, bottom_right, bottom_left]))

def get_sbd(input_img):
    input_img = su.get_bigest_frame('sbdinput', input_img)
    su.export_img('sbd_input.png', input_img)
    sbd = get_sbd_detail('sbd', input_img)
    return ''.join(sbd)

def get_exam_code(input_img):
    input_img = su.get_bigest_frame('exam_code_input', input_img)
    su.export_img('exam_code_input.png', input_img)
    exam_code = get_exam_code_detail('exam_code', input_img)
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

    return sbd

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


# success_img = ['3', '6', '7', '9', '12', '17', '19', '21', '24', '26']

# for img_name in success_img:
# 	path_img = 'E:\\hgedu-test-2\\success\\' + img_name + '.jpg'#'E:\\hgedu-test\\kt1.png'
# 	scan_exam(path_img)

#path_img = 'E:\\hgedu-test-2\\' + 'b5' + '.jpg'#'E:\\hgedu-test\\kt1.png'
#path_img = 'E:\\hgedu-test\\' + 'kt4' + '.png'#'E:\\hgedu-test\\kt1.png'
#path_img = 'E:\\hgedu-test-2\\failed-mo\\' + '20' + '.jpg'#'E:\\hgedu-test\\kt1.png'
#path_img = 'E:\\hgedu-test-2\\failed-fix\\' + '13' + '.jpg'#'E:\\hgedu-test\\kt1.png'
#path_img = 'E:\\hgedu-test-2\\' + 'x6' + '.jpg'#'E:\\hgedu-test\\kt1.png'
#path_img = 'E:\\hgedu-test-5\\12.jpg'
#scan_exam(path_img)
