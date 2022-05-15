from ast import If
from msilib.schema import Directory
import cv2
import numpy as np
from skimage.util import random_noise
import random
import os
import argparse
import json


def apply_motion_blur(image, size=15, angle=0):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    return cv2.filter2D(image, -1, k)


def apply_sandp(image, amount=0.1):
    return random_noise(image, mode='s&p', amount=amount)*255


def change_brightness(image, amount = 10):
    hls = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)
    if amount > 0:
        lim = 255 - amount
        l[l > lim] = 255
        l[l <= lim] += amount
    elif amount < 0:
        amount *= -1
        l[l <= amount] = 0
        l[l > amount] -= amount
    final_hsv = cv2.merge((h, l, s))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HLS2RGB)
    return img

def guassian_blur(image,kernal ,sigma):
    while True:
        try:
            img = cv2.GaussianBlur(image,(kernal,kernal),sigma,cv2.BORDER_DEFAULT)
            break
        except:
            kernal += 1
    return img
    
def guassian_noise(image,sigma): 
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    img = image + gauss
    return img

def change_hue(image, amount = 10):
    hls = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)
    if amount > 0:
        h += amount
    elif amount < 0:
        amount *= -1
        h -= amount 
    final_hsv = cv2.merge((h, l, s))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HLS2RGB)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='stuff')
    parser.add_argument('--rp', type=str,
                        help='read path')
    parser.add_argument('--wp', type=str,
                        help='write path')
    args = parser.parse_args()
    f = open('setting.json')
    setting = json.load(f)
    os.mkdir(f"./{args.wp}")
    for file in os.listdir(args.rp):
        motion_blur_angle = random.randint(0, 180)
        sandp_amount = random.randint(0, 20)/100
        motion_blur_size = random.randint(1, 30)
        brighness_shift = random.randint(-30, 30)
        guassian_blur_kernal = random.randint(10,20)
        guassian_blur_sigma = random.randint(5, 10)
        guassian_noise_amount = random.randint(1, 20)
        change_hue_amount = random.randint(-20,20)
        im = cv2.imread(os.path.join(args.rp, file))
        if setting["motion_blur"]:
            im = apply_motion_blur(im, size=motion_blur_size,
                                angle=motion_blur_angle)
        if setting["guassian_blur"]:
            im = guassian_blur(im,guassian_blur_kernal,guassian_blur_sigma)
        if setting["guassian_noise"]:
            im = guassian_noise(im, guassian_noise_amount)
        if setting["sandp"]:
            im = apply_sandp(im, sandp_amount)
        if setting["change_hue"]:
            im = change_hue(im, change_hue_amount)
        if setting["brightness_shift"]:
            im = change_brightness(im,brighness_shift)
        cv2.imwrite(os.path.join(args.wp, file), im)
