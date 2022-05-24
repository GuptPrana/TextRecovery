import cv2
from imutils import contours
import numpy as np

print_in_func = False


def adap_th_gaus(image, name):
    image_adaptive_th = 0
    for constant in [15]:
        for block_size in [199]:
            imagename = name+"_adap_th_"+str(block_size)+"_"+str(constant)
            image_adaptive_th = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
            if print_in_func:
                print(image_adaptive_th, imagename)
    return image_adaptive_th


def resize(image):
    im_h, im_w = image.shape
    target_w = 128
    target_h = 16
    scale_w = target_w/im_w
    scale_h = target_h/im_h
    scale = min(scale_w, scale_h)
    width = int(im_w * scale)
    height = int(im_h * scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if (width != target_w):
        padding_amount = int((target_w-width)/2)
        padded = np.pad(resized, [(0, ), (padding_amount, )],
                        mode='constant', constant_values=0)
        if (padded.shape[1] != target_w):

            padded = np.pad(padded, [(0, 0), (0, 1)],
                            mode='constant', constant_values=0)
    if(height != target_h):
        padding_amount = int((target_h-height)/2)
        padded = np.pad(resized, [(padding_amount, ), (0, )],
                        mode='constant', constant_values=0)
        if (padded.shape[0] != target_h):
            padded = np.pad(padded, [(0, 1), (0, 0)],
                            mode='constant', constant_values=0)

    return padded


def find_line(img):
    indices = []
    for i, val in enumerate(img):
        density = sum(val)
        if (density > 1000):
            indices.append(i)
    final = []

    start = None
    end = None
    for num, idx in enumerate(indices):
        if not start:
            start = idx

        if num > 0:
            try:
                if indices[num + 1] - idx == 1:
                    continue
                else:
                    end = idx
            except:
                end = idx

        if start and end:
            final.append([start, end])
            start = None
            end = None
    return final

def word_seperator(image , write_folder):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_size = gray.shape[0]*gray.shape[1]
  thresh = adap_th_gaus(gray, "cringe")
  thresh = 255-thresh
  kernel = np.ones((5, 5), np.uint8)
  dilated = cv2.dilate(thresh, kernel, iterations=2)

  ROI_number = 0
  for line in find_line(dilated):
    line_dilated = dilated[line[0]-1:line[1]+1,:]
    line_thresh = thresh[line[0]-1:line[1]+1,:]
    # cv2.imshow("hello",line_dilated)
    # cv2.waitKey()
    # Find contours, sort from left-to-right, then crop
    cnts = cv2.findContours(line_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    # # Filter using contour area and extract ROI
    for c in cnts:
        area = cv2.contourArea(c)
        if area > image_size / 40000:
            x, y, w, h = cv2.boundingRect(c)
            ROI = line_thresh[y:y+h, x:x+w]
            try:
              resized = resize(ROI)
            except:
              continue
            cv2.imwrite(f'{write_folder}/ROI_{ROI_number}.png', resized)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI_number += 1


