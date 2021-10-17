#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import argparse
from os import walk
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser(
    prog='banana',
    description='Segmenta bananas em imagens coloridas.'
)

parser.add_argument('input_img', type=str, help='caminho da imagem de entrada.')
parser.add_argument('output_img', type=str, help='caminho da imagem de saída (segmentação).')

parser.add_argument('-p', '--plot', action='store_true', dest='plot', help='faz o programa plotar a imagem e o gráfico do modelo de cor dela.')
parser.add_argument('-r', '--reduce', action='store', type=int, dest='plot_reduction', help='fator pelo qual o conjunto de pixls é diminuido para visualização.')

args = parser.parse_args()

if args.plot:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors

banana = cv.imread(args.input_img)
banana = cv.cvtColor(banana, cv.COLOR_BGR2RGB)

if args.plot:
    plt.title(args.input_img)
    plt.imshow(banana)
    plt.show()

hsv_banana = cv.cvtColor(banana, cv.COLOR_RGB2HSV)

if args.plot:
    import plotly.graph_objects as go
    import numpy as np

    h, s, v = cv.split(hsv_banana)

    flatH = [v for i, v in enumerate(h.flatten()) if i % args.plot_reduction == 0]
    flatS = [v for i, v in enumerate(s.flatten()) if i % args.plot_reduction == 0]
    flatV = [v for i, v in enumerate(v.flatten()) if i % args.plot_reduction == 0]
    colors = ["hsv(%d,%d,%d)" % v for i, v in enumerate(zip(flatH,flatS,flatV))]

    fig = go.Figure(data=[go.Scatter3d(
        x=flatH,
        y=flatS,
        z=flatV,
        mode='markers',
        marker=dict(
            size=1,
            color=colors,
        )
    )])

    # tight layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                title="Hue"
            ),
            yaxis=dict(
                title='Saturation'
            ),
            zaxis = dict(
                title='Value'
            )
        )
    )
    fig.show()

# hsv_point1 = ()
# hsv_point2 = ()

# mask = cv2.inRange(hsv_banana, light_orange, dark_orange)

# result = cv2.bitwise_and(banana, banana, mask=mask)

# plt.subplot(1, 2, 1)
# plt.imshow(mask, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(result)
# plt.show()




# PONTOS ANA
# hsv_bounds = {
#     "banana1.png": {
#         "lower": (18, 114, 130),
#         "upper": (31, 224, 253)
#     },
#     "banana2.png": {
#         "lower": (19, 54, 97),
#         "upper": (83, 236, 255)
#     },
#     "banana3.png": {
#         "lower": (19, 12, 134),
#         "upper": (135, 255, 255)
#     },

#     "banana4.png": {
#         "lower": (19 , 12, 98),
#         "upper": (175, 248, 255)
#     },
#     "banana5.png": {
#         "lower": (20, 157, 211),
#         "upper": (160, 255, 255)
#     },
#     "banana6.png": {
#         "lower": (18, 50, 44),
#         "upper": (95, 242, 255)
#     },
#     # "banana7.png": {
#     #     "lower": (13, 148, 145),
#     #     "upper": (27, 227, 255)
#     # }
# }
