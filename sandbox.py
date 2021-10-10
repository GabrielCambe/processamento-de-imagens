#!/usr/bin/python3
import numpy as np
import cv2 as cv

for option in cv.__dict__:
    if 'CMP' in option:
        print(option)