#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import argparse
from os import walk
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser(
    prog='banana',
    description='Segmenta bananas amarelas em imagens coloridas.'
)

parser.add_argument('input_img', type=str, help='caminho da imagem de entrada.')
parser.add_argument('output_img', type=str, help='caminho da imagem de saída (segmentação).')
parser.add_argument('-p', '--plot', action='store_true', dest='plot', help='faz o programa plotar a imagem segmentada e sua máscara.')

args = parser.parse_args()


banana = cv.imread(args.input_img)
hsv_banana = cv.cvtColor(banana, cv.COLOR_BGR2HSV)

hsv_bounds = {
    "banana1.png": {
        "lower": (16, 82, 71),
        "upper": (39, 251, 251)
    },
    "banana2.png": {
        "lower": (14, 56, 116),
        "upper": (28, 181, 242)
    },
    "banana3.png": {
        "lower": (15, 70, 69),
        "upper": (33, 236, 254)
    },
    "banana4.png": {
        "lower": (16, 27, 135),
        "upper": (31, 218, 255)
    },
    "banana5.png": {
        "lower": (15, 92, 203),
        "upper": (30, 223, 255)
    },
    "banana6.png": {
        "lower": (14, 8, 53),
        "upper": (43, 240, 255)
    },
    "banana7.png": {
        "lower": (13, 148, 145),
        "upper": (27, 227, 255)
    }
}


lowerMedian = lambda array: sorted(array)[int(len(array)/2 if len(array)%2==1 else (len(array)-1)/2)]
upperMedian = lambda array: sorted(array)[int(len(array)/2 if len(array)%2==1 else (len(array)+1)/2)]

max = lambda array: sorted(array)[-1]

hsv_lowerbound_options = {"h": [], "s": [], "v": []}
hsv_upperbound_options = {"h": [], "s": [], "v": []}

for filename in hsv_bounds:
    hsv_lowerbound_options["h"].append(hsv_bounds[filename]['lower'][0])
    hsv_lowerbound_options["s"].append(hsv_bounds[filename]['lower'][1])
    hsv_lowerbound_options["v"].append(hsv_bounds[filename]['lower'][2])
    
    hsv_upperbound_options["h"].append(hsv_bounds[filename]['upper'][0])
    hsv_upperbound_options["s"].append(hsv_bounds[filename]['upper'][1])
    hsv_upperbound_options["v"].append(hsv_bounds[filename]['upper'][2])

hsv_lowerbound = (
    lowerMedian(hsv_lowerbound_options["h"]),
    lowerMedian(hsv_lowerbound_options["s"]),
    lowerMedian(hsv_lowerbound_options["v"])
)
hsv_upperbound = (
    upperMedian(hsv_upperbound_options["h"]),
    upperMedian(hsv_upperbound_options["s"]),
    upperMedian(hsv_upperbound_options["v"])
)

# Para usar as bounds de cada arquivo
# hsv_lowerbound = hsv_bounds[args.input_img]['lower']
# hsv_upperbound = hsv_bounds[args.input_img]['upper']

mask = cv.inRange(hsv_banana, hsv_lowerbound, hsv_upperbound)

result = cv.bitwise_and(banana, banana, mask=mask)
cv.imwrite(args.output_img, result)

# Plota a imagem segmntada e a máscara usada lado a lado.
if args.plot:
    from matplotlib import pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()
