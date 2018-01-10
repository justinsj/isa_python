# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 02:12:46 2018

@author: JustinSanJuan
"""

ground_truth_list = [29, 5, 0, 28, 6, 4, 52, 45, 48, 29, 4, 6, 24, 0, 52, 5, 48, 41, 5, 0, 28, 6, 4, 29, 52, 45, 48, 6, 29, 28, 4, 52, 5, 0, 45, 48, 4, 28, 29, 0, 6, 52, 5, 45, 48, 0, 6, 29, 28, 4, 52, 5, 45, 48, 28, 0, 29, 6, 4, 52, 5, 45, 48, 0, 6, 28, 29, 4, 52, 5, 45, 48, 0, 4, 5, 28, 29, 6, 52, 45, 48, 4, 6, 5, 29, 28, 0, 52, 45, 48, 29, 5, 45, 28, 0, 60, 6, 4, 52, 48, 61, 60, 61, 4, 29, 48, 5, 56, 45, 0, 28, 56, 6, 52, 29, 0, 5, 28, 45, 4, 6, 56, 56, 52, 48, 29, 4, 5, 45, 28, 0, 56, 56, 6, 52, 48, 4, 28, 6, 5, 0, 56, 56, 52, 45, 48, 29, 6, 0, 45, 5, 48, 4, 29, 28, 56, 56, 52, 4, 28, 5, 45, 0, 29, 56, 56, 52, 48, 6, 29, 5, 4, 0, 6, 56, 28, 56, 52, 45, 48, 6, 28, 45, 29, 4, 0, 5, 56, 56, 52, 48, 6, 0, 45, 28, 29, 5, 4, 56, 56, 52, 48, 28, 29, 0, 4, 48, 5, 61, 45, 60, 61, 60, 6, 52, 5, 45, 0, 4, 6, 29, 28, 48, 61, 60, 61, 60, 52, 5, 28, 29, 48, 4, 61, 0, 60, 61, 60, 52, 45, 6, 48, 28, 5, 0, 6, 45, 60, 29, 4, 61, 60, 52, 61, 29, 48, 45, 0, 5, 28, 4, 60, 61, 60, 6, 52, 61, 5, 48, 45, 0, 4, 28, 60, 61, 6, 60, 52, 61, 29, 28, 4, 29, 45, 60, 0, 61, 5, 48, 52, 6, 61, 60, 48, 29, 61, 28, 5, 0, 4, 52, 60, 61, 60, 6, 45, 28, 4, 5, 60, 48, 0, 29, 6, 61, 45, 52, 61, 60, 60, 48, 29, 6, 4, 0, 45, 28, 5, 52, 61, 60, 61, 60, 0, 5, 4, 61, 29, 6, 28, 45, 48, 52, 61, 60, 4, 60, 28, 5, 61, 48, 0, 29, 45, 52, 61, 60, 6, 28, 6, 45, 5, 0, 29, 4, 48, 60, 52, 61, 60, 61, 4, 28, 45, 5, 61, 60, 48, 6, 0, 29, 52, 61, 60, 61, 45, 29, 60, 4, 5, 28, 48, 0, 6, 52, 61, 60, 48, 52, 6, 5, 45, 28, 0, 4, 29, 56, 56, 0, 60, 61, 48, 29, 4, 28, 5, 52, 6, 61, 60, 45, 28, 52, 6, 0, 5, 29, 48, 61, 45, 4, 60, 61, 60, 60, 4, 52, 48, 29, 0, 5, 28, 61, 6, 61, 60, 45, 60, 6, 29, 0, 52, 45, 5, 4, 28, 48, 61, 60, 61, 45, 29, 0, 6, 4, 4, 52, 48, 4, 29, 6, 45, 0, 4, 52, 48, 0, 4, 6, 4, 29, 52, 48, 45, 45, 4, 0, 29, 4, 6, 52, 48, 4, 45, 4, 29, 0, 52, 48, 6, 0, 4, 29, 45, 6, 4, 52, 48, 6, 0, 45, 4, 4, 48, 52, 29, 4, 0, 29, 45, 6, 4, 52, 48, 4, 45, 6, 0, 4, 29, 52, 48, 45, 4, 6, 29, 4, 0, 48, 52, 48, 6, 29, 4, 52, 45, 4, 0, 6, 29, 48, 45, 4, 52, 4, 0, 48, 45, 6, 29, 52, 4, 0, 4, 48, 4, 6, 29, 52, 45, 4, 0, 48, 6, 45, 29, 52, 4, 4, 0, 4, 29, 48, 6, 45, 52, 4, 0, 45, 48, 4, 29, 6, 52, 4, 0, 29, 48, 45, 6, 4, 52, 4, 0, 6, 4, 48, 29, 45, 52, 4, 0, 4, 52, 48, 45, 4, 0, 60, 6, 61, 60, 61, 29, 60, 6, 4, 29, 48, 52, 45, 4, 0, 61, 60, 61, 4, 29, 48, 52, 45, 4, 0, 6, 60, 61, 60, 61, 6, 4, 48, 52, 45, 4, 0, 60, 61, 60, 61, 29, 4, 60, 48, 52, 45, 4, 0, 6, 61, 60, 29, 61, 29, 6, 61, 4, 45, 48, 52, 4, 0, 60, 61, 60, 4, 29, 48, 52, 45, 4, 0, 60, 6, 61, 60, 61, 29, 6, 61, 4, 48, 52, 45, 60, 4, 0, 61, 60, 4, 6, 61, 48, 52, 45, 4, 0, 29, 61, 60, 60, 45, 6, 48, 29, 56, 56, 4, 4, 0, 52, 29, 56, 6, 56, 4, 45, 52, 48, 4, 0, 29, 56, 6, 56, 4, 45, 52, 48, 4, 0, 29, 56, 56, 6, 4, 48, 52, 45, 4, 0, 48, 29, 56, 56, 6, 4, 45, 52, 4, 0, 48, 29, 6, 45, 56, 56, 4, 52, 4, 0, 29, 48, 45, 56, 56, 6, 4, 52, 4, 0, 29, 56, 56, 6, 4, 52, 45, 48, 4, 0, 45, 48, 29, 4, 56, 56, 6, 52, 4, 0, 45, 29, 6, 56, 56, 4, 48, 52, 4, 0, 45, 5, 0, 3, 29, 45, 52, 11, 4, 48, 2, 5, 3, 2, 5, 45, 0, 5, 4, 45, 52, 48, 11, 29, 3, 5, 45, 0, 45, 52, 11, 48, 2, 5, 29, 4, 5, 2, 11, 0, 45, 5, 3, 4, 45, 52, 48, 29, 0, 4, 45, 2, 5, 5, 3, 29, 45, 11, 52, 48, 2, 3, 45, 29, 5, 5, 0, 45, 52, 48, 11, 4, 45, 29, 3, 5, 45, 52, 11, 4, 0, 48, 2, 5, 45, 29, 5, 2, 45, 3, 5, 52, 11, 4, 0, 48, 5, 0, 5, 29, 2, 45, 3, 45, 52, 11, 4, 48, 5, 5, 2, 3, 11, 45, 45, 52, 4, 0, 48, 29, 0, 48, 45, 45, 2, 5, 4, 5, 3, 29, 11, 52, 0, 5, 29, 4, 2, 5, 11, 45, 3, 52, 45, 48, 4, 2, 0, 48, 5, 45, 11, 29, 3, 5, 52, 45, 0, 3, 5, 45, 5, 2, 45, 29, 11, 4, 52, 48, 11, 29, 4, 5, 2, 45, 0, 45, 3, 48, 5, 52, 0, 45, 4, 29, 5, 11, 2, 5, 3, 52, 45, 48, 29, 2, 5, 11, 4, 5, 0, 3, 45, 52, 45, 48, 2, 45, 0, 3, 5, 29, 5, 4, 11, 52, 45, 48, 11, 2, 3, 0, 45, 5, 29, 5, 45, 48, 4, 52, 29, 2, 45, 0, 11, 48, 5, 4, 3, 5, 45, 52, 3, 45, 5, 11, 2, 45, 5, 48, 56, 56, 52, 29, 4, 0, 3, 5, 2, 5, 45, 48, 45, 11, 56, 56, 52, 29, 4, 0, 3, 45, 5, 5, 48, 56, 56, 45, 52, 29, 11, 2, 4, 0, 3, 5, 2, 45, 48, 5, 56, 56, 52, 11, 29, 45, 4, 0, 5, 3, 48, 5, 45, 45, 2, 56, 56, 52, 11, 4, 0, 29, 3, 5, 45, 11, 45, 5, 48, 56, 56, 52, 29, 4, 0, 2, 56, 45, 2, 45, 5, 11, 48, 5, 3, 56, 52, 29, 4, 0, 45, 11, 29, 2, 5, 3, 5, 45, 56, 56, 52, 48, 4, 0, 2, 5, 5, 3, 45, 11, 29, 56, 56, 52, 45, 48, 4, 0, 5, 45, 11, 2, 56, 3, 5, 45, 29, 48, 56, 52, 4, 0, 3, 48, 45, 0, 29, 4, 5, 45, 56, 56, 52, 11, 2, 5, 5, 4, 3, 0, 45, 45, 56, 56, 52, 11, 48, 2, 5, 29, 45, 0, 5, 4, 45, 3, 56, 56, 52, 11, 48, 2, 5, 29, 3, 45, 4, 5, 0, 45, 29, 56, 56, 52, 11, 48, 2, 5, 45, 4, 0, 3, 45, 5, 56, 56, 52, 11, 48, 2, 5, 29, 3, 4, 5, 0, 45, 45, 56, 56, 52, 11, 48, 2, 5, 29, 0, 45, 3, 5, 4, 45, 56, 56, 52, 11, 48, 2, 5, 29, 45, 5, 3, 45, 29, 4, 0, 56, 56, 52, 11, 48, 2, 5, 4, 45, 3, 0, 29, 5, 45, 56, 56, 52, 11, 48, 2, 5, 4, 45, 3, 45, 5, 29, 0, 56, 56, 52, 11, 48, 2, 5, 6, 45, 11, 4, 0, 2, 29, 4, 11, 28, 5, 52, 11, 29, 45, 5, 11, 0, 6, 28, 4, 2, 52, 4, 29, 45, 28, 11, 2, 0, 11, 4, 6, 5, 4, 52, 4, 0, 11, 28, 11, 2, 45, 6, 29, 5, 52, 4, 4, 2, 4, 28, 0, 45, 11, 11, 6, 29, 5, 52, 11, 2, 0, 11, 28, 4, 6, 45, 29, 5, 52, 4, 4, 4, 28, 29, 0, 6, 11, 2, 5, 45, 52, 11, 5, 11, 2, 45, 11, 29, 6, 56, 56, 28, 52, 4, 0, 4, 29, 2, 11, 5, 6, 45, 28, 11, 56, 56, 52, 4, 0, 4, 5, 4, 28, 45, 11, 29, 2, 11, 6, 56, 56, 52, 4, 0, 45, 5, 6, 11, 28, 4, 11, 2, 56, 56, 52, 29, 4, 0, 6, 11, 2, 29, 11, 5, 45, 56, 56, 4, 28, 52, 4, 0, 11, 11, 29, 4, 6, 2, 28, 5, 56, 56, 52, 4, 0, 45, 4, 5, 6, 45, 11, 28, 11, 2, 56, 56, 52, 29, 4, 0, 6, 2, 29, 5, 11, 45, 11, 28, 56, 56, 52, 4, 4, 0, 6, 11, 11, 5, 4, 29, 28, 45, 2, 56, 56, 52, 4, 0, 5, 4, 6, 28, 11, 45, 11, 29, 2, 56, 56, 52, 4, 0, 45, 5, 28, 4, 29, 2, 11, 56, 56, 52, 6, 4, 0, 11, 5, 45, 11, 4, 2, 28, 29, 56, 56, 52, 6, 4, 0, 11, 45, 2, 28, 11, 4, 5, 29, 56, 56, 52, 6, 4, 0, 11, 4, 45, 6, 28, 5, 11, 29, 2, 56, 56, 52, 11, 4, 0, 4, 45, 5, 29, 2, 28, 56, 56, 52, 6, 11, 11, 4, 0, 11, 4, 29, 28, 5, 45, 2, 56, 56, 52, 6, 11, 4, 0, 28, 45, 5, 4, 11, 2, 29, 56, 56, 52, 11, 4, 0, 6, 29, 11, 2, 4, 45, 5, 28, 56, 56, 52, 11, 6, 4, 0, 4, 28, 5, 45, 29, 2, 6, 11, 56, 56, 52, 11, 4, 0, 45, 11, 6, 2, 4, 29, 28, 56, 56, 52, 5, 11, 4, 0, 45, 11, 6, 5, 4, 11, 2, 56, 56, 28, 29, 52, 4, 0, 5, 11, 2, 6, 11, 4, 45, 0, 56, 56, 28, 29, 52, 4, 2, 11, 11, 6, 5, 45, 0, 4, 56, 56, 52, 28, 29, 4, 5, 45, 11, 0, 4, 11, 2, 6, 56, 56, 52, 28, 29, 4, 11, 6, 4, 2, 45, 0, 11, 5, 4, 56, 56, 28, 52, 29, 11, 6, 5, 45, 4, 2, 11, 56, 56, 52, 28, 29, 4, 0, 11, 6, 4, 11, 0, 5, 45, 4, 2, 56, 56, 28, 52, 29, 0, 6, 2, 11, 45, 5, 11, 4, 56, 56, 52, 28, 29, 4, 45, 11, 2, 6, 0, 11, 5, 4, 56, 56, 28, 29, 52, 4, 5, 0, 11, 6, 4, 45, 2, 11, 56, 56, 28, 29, 52, 4, 48, 5, 0, 5, 45, 28, 8, 52, 45, 5, 8, 48, 5, 0, 28, 45, 52, 45, 52, 8, 45, 0, 48, 5, 5, 45, 28, 45, 8, 45, 0, 5, 5, 28, 48, 52, 45, 5, 0, 5, 28, 45, 48, 8, 52, 0, 8, 5, 45, 5, 45, 28, 48, 52, 45, 5, 45, 28, 5, 48, 8, 0, 52, 28, 0, 48, 5, 5, 45, 45, 8, 52, 5, 8, 0, 5, 28, 48, 45, 52, 45, 45, 5, 28, 0, 48, 8, 5, 45, 52, 5, 28, 45, 5, 52, 45, 48, 8, 0, 45, 28, 45, 5, 5, 52, 48, 8, 0, 28, 45, 5, 5, 45, 52, 48, 8, 0, 28, 5, 45, 45, 5, 52, 48, 8, 0, 45, 28, 5, 5, 45, 52, 48, 8, 0, 45, 5, 5, 28, 45, 52, 48, 8, 0, 5, 5, 28, 45, 52, 45, 48, 8, 0, 5, 45, 28, 5, 52, 45, 48, 8, 0, 5, 45, 45, 5, 28, 52, 48, 8, 0, 5, 28, 45, 45, 5, 52, 48, 56, 56, 8, 0, 45, 45, 5, 28, 5, 52, 48, 56, 56, 8, 0, 48, 45, 28, 5, 45, 52, 8, 0, 56, 56, 5, 45, 28, 5, 45, 56, 56, 48, 5, 52, 8, 0, 45, 5, 28, 45, 52, 8, 0, 56, 56, 5, 48, 5, 45, 28, 45, 52, 48, 8, 0, 56, 5, 56, 45, 45, 5, 28, 5, 52, 8, 0, 56, 56, 48, 5, 45, 45, 28, 52, 48, 56, 56, 5, 8, 0, 45, 45, 28, 5, 5, 52, 8, 0, 48, 56, 56, 28, 45, 45, 5, 52, 48, 8, 0, 56, 56, 5, 48, 5, 28, 5, 45, 45, 52, 8, 0, 56, 56, 45, 48, 45, 28, 5, 5, 52, 8, 0, 56, 56, 5, 45, 45, 5, 28, 52, 48, 8, 0, 56, 56, 5, 28, 45, 5, 45, 52, 48, 8, 0, 56, 56, 28, 5, 45, 5, 45, 52, 48, 8, 0, 56, 56, 28, 45, 5, 5, 45, 52, 48, 8, 0, 56, 56, 28, 5, 5, 48, 45, 45, 52, 8, 0, 56, 56, 45, 5, 45, 28, 5, 52, 48, 8, 0, 56, 56, 45, 5, 45, 28, 5, 52, 48, 8, 0, 56, 56, 5, 45, 45, 5, 28, 52, 48, 8, 0, 56, 56, 6, 28, 0, 0, 3, 4, 6, 52, 45, 45, 48, 48, 6, 28, 45, 3, 4, 0, 0, 6, 52, 45, 48, 48, 3, 6, 0, 6, 28, 4, 45, 0, 52, 45, 48, 48, 0, 0, 6, 45, 3, 4, 45, 28, 52, 48, 48, 6, 45, 0, 6, 0, 4, 6, 28, 52, 45, 48, 48, 3, 3, 6, 0, 45, 6, 4, 0, 28, 52, 45, 48, 48, 0, 28, 3, 0, 4, 6, 6, 52, 45, 45, 48, 48, 6, 4, 0, 28, 6, 0, 3, 52, 45, 45, 48, 48, 4, 45, 28, 0, 3, 6, 0, 6, 45, 52, 48, 48, 28, 45, 0, 6, 6, 4, 0, 52, 45, 48, 48, 3, 3, 28, 0, 4, 48, 45, 0, 6, 52, 48, 6, 56, 56, 45, 6, 0, 0, 45, 3, 45, 4, 48, 28, 52, 56, 56, 6, 48, 0, 6, 4, 45, 0, 6, 48, 45, 3, 28, 56, 56, 48, 52, 45, 0, 45, 4, 3, 28, 0, 6, 48, 56, 56, 6, 48, 52, 45, 4, 45, 0, 28, 6, 0, 6, 3, 48, 56, 56, 48, 52, 4, 45, 48, 6, 0, 6, 45, 0, 3, 56, 56, 28, 48, 52, 4, 3, 6, 45, 28, 48, 0, 0, 45, 6, 56, 56, 48, 52, 0, 56, 3, 6, 45, 4, 48, 6, 45, 28, 0, 56, 52, 48, 6, 45, 0, 3, 6, 48, 4, 0, 28, 56, 56, 52, 45, 48, 3, 6, 4, 45, 0, 6, 0, 45, 56, 56, 28, 52, 48, 48, 45, 45, 6, 28, 56, 56, 52, 48, 48, 3, 0, 6, 0, 4, 45, 28, 6, 4, 45, 56, 56, 52, 48, 48, 3, 0, 6, 0, 45, 45, 28, 6, 4, 56, 56, 52, 48, 48, 3, 0, 6, 0, 45, 48, 45, 28, 6, 56, 56, 4, 52, 48, 3, 0, 6, 0, 4, 28, 6, 45, 56, 56, 52, 48, 48, 3, 0, 6, 0, 45, 45, 28, 3, 45, 0, 56, 56, 6, 52, 48, 48, 6, 0, 4, 45, 28, 45, 6, 56, 56, 4, 52, 48, 48, 3, 0, 6, 0, 6, 45, 45, 56, 56, 28, 52, 48, 48, 3, 0, 6, 0, 4, 45, 45, 28, 56, 56, 6, 4, 52, 48, 48, 3, 0, 6, 0, 4, 6, 45, 45, 28, 56, 56, 52, 48, 48, 3, 0, 6, 0, 0, 48, 4, 45, 3, 45, 28, 56, 56, 6, 48, 6, 0, 52, 3, 45, 52, 45, 48, 4, 28, 0, 48, 56, 56, 6, 6, 0, 45, 45, 28, 3, 6, 48, 0, 4, 56, 56, 52, 48, 6, 0, 4, 28, 45, 6, 3, 48, 0, 56, 56, 52, 48, 45, 6, 0, 45, 0, 28, 3, 48, 4, 56, 56, 52, 6, 48, 45, 6, 0, 45, 48, 3, 45, 4, 28, 0, 56, 56, 6, 48, 6, 0, 52, 48, 6, 4, 45, 0, 28, 45, 56, 56, 48, 3, 6, 0, 52, 6, 48, 28, 4, 3, 0, 45, 56, 56, 52, 45, 48, 6, 0, 48, 52, 28, 4, 0, 45, 3, 56, 56, 6, 48, 45, 6, 0, 11, 29, 1, 45, 6, 45, 6, 48, 56, 56, 52, 3, 0, 2, 0, 6, 11, 45, 29, 45, 56, 56, 6, 48, 3, 0, 1, 52, 2, 0, 11, 6, 29, 6, 45, 45, 56, 56, 52, 3, 0, 48, 1, 2, 0, 45, 29, 45, 11, 56, 56, 6, 6, 52, 48, 3, 0, 1, 2, 0, 1, 45, 48, 11, 6, 45, 56, 56, 6, 3, 0, 52, 29, 2, 0, 6, 6, 45, 45, 11, 56, 56, 52, 48, 3, 0, 29, 1, 2, 0, 45, 6, 11, 6, 45, 29, 1, 56, 56, 52, 48, 3, 0, 2, 0, 45, 6, 29, 6, 48, 56, 56, 45, 11, 52, 3, 0, 1, 2, 0, 48, 45, 6, 6, 29, 11, 45, 56, 56, 52, 3, 0, 1, 2, 0, 6, 6, 29, 56, 56, 45, 45, 52, 48, 11, 3, 0, 1, 2, 0, 29, 6, 1, 45, 45, 52, 6, 48, 3, 0, 2, 0, 11, 11, 6, 1, 6, 45, 29, 45, 52, 48, 3, 0, 2, 0, 6, 45, 11, 1, 6, 29, 52, 45, 48, 3, 0, 2, 0, 29, 6, 45, 45, 6, 52, 48, 3, 0, 11, 1, 2, 0, 45, 45, 1, 6, 29, 6, 52, 48, 3, 0, 11, 2, 0, 6, 45, 45, 29, 1, 6, 52, 48, 3, 0, 11, 2, 0, 29, 11, 45, 6, 1, 45, 52, 48, 3, 0, 6, 2, 0, 1, 29, 45, 6, 6, 45, 52, 48, 3, 0, 11, 2, 0, 45, 29, 11, 1, 6, 45, 6, 52, 52, 3, 0, 2, 0, 1, 45, 6, 3, 45, 56, 29, 0, 56, 6, 52, 48, 11, 2, 0, 29, 1, 45, 3, 0, 45, 6, 56, 56, 6, 48, 52, 11, 2, 0, 1, 3, 6, 29, 45, 0, 11, 56, 56, 6, 45, 52, 48, 2, 0, 45, 0, 3, 11, 1, 6, 29, 45, 56, 56, 6, 52, 48, 2, 0, 45, 3, 1, 11, 45, 0, 29, 6, 56, 56, 52, 48, 2, 0, 6, 45, 3, 0, 29, 1, 6, 6, 11, 56, 56, 45, 52, 48, 2, 0, 1, 6, 3, 29, 45, 0, 56, 56, 6, 45, 52, 48, 11, 2, 0, 29, 0, 3, 1, 6, 45, 45, 6, 56, 56, 52, 48, 11, 2, 0, 45, 45, 6, 1, 0, 29, 3, 56, 56, 52, 48, 2, 0, 11, 6, 4, 45, 2, 5, 45, 4, 4, 45, 0, 11, 0, 11, 52, 1, 0, 2, 4, 0, 4, 11, 45, 45, 11, 52, 45, 1, 4, 5, 4, 0, 11, 45, 0, 4, 5, 45, 2, 11, 52, 4, 45, 1, 11, 45, 4, 4, 0, 4, 45, 5, 0, 2, 11, 52, 45, 1, 2, 11, 45, 11, 0, 45, 4, 0, 4, 45, 52, 4, 5, 1, 45, 2, 4, 11, 4, 4, 5, 11, 0, 0, 52, 45, 45, 1, 4, 5, 11, 2, 45, 0, 11, 45, 4, 0, 4, 52, 45, 1, 0, 2, 45, 4, 11, 5, 45, 11, 45, 0, 4, 4, 1, 52, 5, 11, 0, 2, 45, 4, 4, 4, 45, 0, 45, 11, 52, 1, 45, 4, 2, 0, 0, 45, 5, 4, 4, 1, 45, 52, 11, 11, 4, 1, 45, 4, 0, 5, 45, 2, 4, 0, 45, 52, 11, 11, 0, 1, 45, 2, 4, 0, 45, 4, 5, 4, 45, 52, 11, 11, 4, 45, 1, 4, 60, 4, 61, 2, 45, 45, 5, 52, 11, 11, 61, 60, 0, 0, 4, 60, 2, 61, 5, 4, 45, 45, 1, 4, 61, 60, 52, 11, 11, 0, 0, 45, 4, 1, 60, 61, 4, 45, 61, 60, 52, 45, 45, 2, 4, 5, 11, 0, 0, 11, 4, 4, 1, 4, 61, 45, 5, 60, 61, 60, 52, 45, 45, 11, 0, 0, 11, 2, 4, 45, 4, 61, 60, 1, 45, 61, 60, 52, 45, 2, 4, 5, 11, 0, 0, 11, 45, 61, 1, 4, 4, 60, 45, 4, 52, 45, 61, 60, 2, 5, 11, 0, 0, 11, 4, 1, 5, 4, 2, 4, 61, 60, 45, 45, 61, 60, 52, 11, 0, 0, 11, 45, 5, 4, 61, 4, 11, 60, 45, 4, 45, 2, 1, 45, 52, 61, 60, 0, 0, 11, 1, 5, 61, 45, 4, 4, 45, 60, 4, 61, 60, 52, 45, 2, 11, 0, 0, 11, 4, 1, 60, 45, 61, 2, 60, 4, 4, 45, 61, 11, 5, 45, 52, 11, 0, 0, 4, 4, 11, 4, 45, 2, 60, 45, 5, 0, 61, 45, 61, 60, 52, 11, 1, 0, 5, 45, 4, 61, 45, 2, 60, 4, 1, 4, 45, 61, 60, 52, 11, 11, 0, 0, 45, 4, 11, 4, 61, 4, 5, 60, 2, 45, 45, 52, 61, 60, 11, 1, 0, 0, 4, 45, 4, 61, 5, 45, 2, 4, 60, 1, 52, 61, 60, 11, 11, 0, 0, 45, 5, 4, 4, 45, 2, 60, 61, 4, 45, 52, 61, 60, 45, 11, 11, 1, 0, 0, 4, 4, 5, 45, 61, 2, 45, 60, 4, 1, 61, 60, 52, 11, 45, 11, 0, 0, 60, 5, 2, 4, 61, 45, 4, 45, 45, 1, 4, 61, 60, 52, 11, 11, 0, 0, 4, 61, 45, 45, 2, 5, 60, 4, 0, 4, 61, 60, 52, 45, 11, 11, 1, 0, 1, 5, 45, 2, 60, 45, 4, 4, 45, 4, 61, 61, 60, 52, 11, 11, 0, 0, 48, 3, 5, 4, 8, 29, 45, 28, 52, 11, 7, 0, 8, 3, 28, 5, 4, 11, 48, 29, 45, 52, 7, 0, 45, 48, 3, 4, 8, 11, 5, 29, 28, 52, 7, 0, 48, 8, 45, 4, 28, 3, 29, 5, 11, 52, 7, 0, 3, 48, 11, 28, 4, 29, 5, 45, 8, 52, 7, 0, 48, 5, 29, 4, 45, 3, 8, 28, 52, 11, 7, 0, 4, 45, 29, 8, 5, 48, 3, 28, 52, 11, 7, 0, 48, 3, 45, 4, 11, 8, 5, 28, 29, 52, 7, 0, 28, 48, 4, 45, 8, 3, 29, 5, 52, 11, 7, 0, 29, 11, 3, 5, 45, 8, 48, 4, 28, 52, 7, 0, 4, 8, 56, 45, 28, 56, 29, 5, 52, 11, 7, 0, 48, 3, 56, 28, 29, 45, 3, 5, 8, 4, 56, 52, 7, 0, 11, 48, 56, 5, 28, 56, 8, 52, 45, 3, 7, 0, 11, 48, 4, 29, 8, 56, 28, 56, 5, 45, 52, 7, 0, 11, 48, 4, 29, 3, 56, 56, 29, 5, 28, 8, 3, 52, 45, 7, 0, 11, 48, 4, 56, 56, 3, 5, 28, 8, 29, 52, 7, 0, 11, 45, 48, 4, 56, 56, 3, 28, 5, 8, 45, 52, 7, 0, 11, 48, 4, 29, 3, 8, 56, 56, 4, 5, 28, 52, 7, 0, 45, 11, 48, 29, 56, 56, 28, 8, 5, 52, 3, 45, 7, 0, 11, 48, 4, 29, 3, 8, 4, 5, 29, 28, 52, 45, 56, 56, 11, 7, 0, 48, 4, 28, 3, 8, 5, 52, 56, 56, 45, 11, 7, 0, 48, 29, 8, 5, 28, 4, 3, 29, 56, 56, 52, 11, 7, 0, 45, 48, 5, 3, 29, 28, 4, 8, 56, 56, 52, 11, 7, 0, 48, 45, 4, 3, 28, 8, 5, 56, 56, 45, 52, 11, 7, 0, 48, 29, 3, 4, 5, 45, 29, 8, 56, 56, 28, 52, 11, 7, 0, 48, 3, 29, 4, 45, 5, 28, 8, 56, 56, 52, 11, 7, 0, 48, 28, 5, 3, 4, 29, 8, 45, 56, 56, 52, 11, 7, 0, 48, 28, 4, 29, 45, 3, 5, 8, 56, 56, 52, 11, 7, 0, 48, 4, 45, 28, 29, 3, 5, 8, 56, 56, 52, 11, 7, 0, 48, 28, 8, 3, 11, 56, 56, 52, 45, 48, 4, 5, 29, 7, 0, 11, 8, 28, 3, 56, 56, 52, 45, 48, 4, 5, 29, 7, 0, 8, 28, 3, 11, 56, 56, 52, 45, 48, 4, 5, 7, 0, 29, 8, 29, 3, 48, 28, 56, 56, 52, 45, 11, 4, 5, 7, 0, 28, 3, 7, 8, 29, 56, 56, 52, 45, 11, 48, 4, 5, 0, 7, 29, 45, 28, 3, 8, 56, 56, 52, 11, 48, 4, 5, 0, 29, 3, 8, 7, 56, 56, 28, 52, 45, 11, 48, 4, 5, 0, 7, 3, 28, 11, 8, 56, 56, 52, 45, 48, 4, 5, 0, 29, 11, 8, 45, 29, 28, 7, 3, 56, 56, 52, 48, 4, 5, 0, 11, 7, 3, 28, 29, 8, 56, 56, 52, 45, 48, 4, 5, 0, 60, 0, 61, 45, 2, 60, 45, 5, 61, 5, 52, 48, 11, 2, 0, 11, 0, 2, 5, 61, 45, 0, 60, 60, 5, 11, 61, 45, 48, 11, 52, 2, 0, 2, 0, 61, 60, 0, 5, 2, 11, 5, 45, 45, 60, 61, 52, 48, 11, 2, 0, 2, 0, 2, 0, 45, 61, 11, 5, 61, 45, 60, 5, 60, 52, 48, 11, 2, 0, 2, 0, 2, 60, 0, 5, 45, 5, 61, 61, 11, 60, 11, 45, 52, 48, 2, 0, 2, 0, 45, 60, 5, 61, 45, 0, 5, 2, 11, 60, 61, 52, 48, 11, 2, 0, 2, 0, 60, 5, 0, 60, 5, 61, 2, 61, 45, 11, 45, 52, 48, 11, 2, 0, 2, 0, 60, 5, 0, 45, 5, 45, 61, 2, 61, 60, 48, 52, 11, 11, 2, 0, 2, 0, 61, 45, 2, 61, 5, 0, 60, 11, 5, 45, 60, 52, 48, 11, 2, 0, 2, 0, 61, 5, 45, 11, 11, 60, 5, 48, 61, 60, 2, 0, 2, 0, 2, 0, 45, 52, 11, 5, 11, 45, 48, 52, 60, 61, 5, 61, 60, 45, 2, 0, 2, 0, 2, 0, 5, 61, 11, 11, 48, 45, 52, 5, 60, 61, 60, 2, 0, 2, 0, 2, 0, 45, 60, 45, 61, 5, 11, 52, 11, 5, 48, 61, 60, 45, 2, 0, 2, 0, 2, 0, 48, 60, 5, 45, 61, 11, 45, 5, 11, 52, 61, 60, 2, 0, 2, 0, 2, 0, 60, 5, 11, 52, 48, 11, 5, 45, 61, 61, 60, 2, 0, 2, 0, 2, 0, 45, 48, 11, 52, 5, 11, 45, 45, 2, 0, 2, 0, 2, 0, 5, 48, 52, 11, 5, 5, 11, 45, 45, 2, 0, 2, 0, 2, 0, 52, 5, 11, 48, 11, 5, 45, 45, 2, 0, 2, 0, 2, 0]
prediction_list = [29, 5, 0, 28, 23, 23, 52, 41, 52, 29, 23, 23, 24, 0, 52, 3, 52, 41, 5, 0, 28, 23, 23, 29, 52, 45, 52, 23, 29, 28, 23, 52, 3, 0, 41, 52, 23, 28, 29, 0, 23, 52, 5, 41, 52, 0, 23, 29, 24, 23, 52, 5, 45, 52, 24, 0, 29, 23, 23, 23, 5, 41, 23, 0, 23, 24, 29, 23, 52, 5, 45, 52, 0, 23, 3, 28, 29, 23, 52, 41, 52, 23, 23, 5, 29, 28, 0, 52, 45, 52, 29, 5, 45, 28, 0, 13, 23, 10, 23, 52, 52, 23, 23, 4, 29, 52, 5, 52, 44, 0, 28, 52, 23, 52, 29, 0, 5, 28, 44, 4, 23, 52, 52, 52, 52, 29, 23, 5, 45, 28, 0, 52, 23, 23, 52, 52, 4, 28, 23, 5, 0, 52, 52, 52, 41, 52, 29, 23, 0, 45, 5, 52, 7, 29, 28, 52, 52, 52, 4, 28, 5, 45, 0, 29, 52, 52, 52, 52, 23, 29, 5, 4, 0, 23, 52, 28, 52, 23, 41, 52, 23, 28, 41, 29, 23, 0, 5, 23, 52, 52, 52, 23, 0, 44, 28, 29, 5, 4, 52, 52, 52, 52, 28, 29, 0, 7, 27, 5, 12, 45, 13, 12, 13, 37, 52, 5, 45, 23, 4, 23, 29, 28, 27, 12, 13, 12, 13, 52, 5, 28, 29, 27, 7, 12, 23, 13, 12, 13, 52, 45, 37, 27, 28, 5, 0, 23, 45, 13, 12, 4, 12, 13, 52, 12, 29, 27, 45, 23, 5, 28, 7, 13, 12, 13, 23, 23, 12, 5, 27, 45, 23, 4, 28, 52, 12, 23, 13, 23, 12, 12, 28, 23, 29, 45, 13, 23, 12, 5, 27, 52, 6, 12, 13, 27, 29, 23, 28, 5, 0, 4, 23, 13, 12, 13, 23, 41, 28, 7, 5, 52, 27, 23, 29, 23, 12, 45, 23, 12, 13, 13, 27, 29, 23, 4, 23, 45, 28, 5, 52, 52, 13, 12, 13, 0, 5, 23, 12, 29, 23, 28, 45, 27, 52, 12, 13, 11, 13, 28, 5, 12, 27, 0, 29, 13, 52, 12, 13, 23, 28, 23, 45, 5, 0, 29, 11, 27, 13, 52, 12, 13, 12, 23, 28, 18, 5, 12, 23, 27, 23, 0, 29, 52, 12, 13, 12, 13, 29, 13, 11, 5, 28, 27, 0, 23, 52, 12, 13, 27, 52, 23, 5, 27, 28, 0, 1, 29, 52, 52, 0, 13, 12, 27, 12, 11, 28, 5, 52, 23, 12, 13, 45, 28, 23, 23, 0, 5, 29, 27, 12, 18, 11, 13, 12, 13, 13, 23, 52, 27, 29, 0, 5, 28, 12, 23, 12, 13, 45, 13, 23, 29, 0, 23, 45, 5, 11, 28, 27, 12, 13, 12, 45, 29, 0, 23, 4, 4, 52, 52, 4, 29, 23, 45, 0, 4, 52, 52, 0, 4, 23, 4, 29, 52, 52, 45, 45, 4, 0, 29, 4, 6, 52, 52, 4, 45, 4, 29, 0, 52, 52, 6, 0, 4, 29, 45, 23, 4, 52, 52, 23, 0, 45, 4, 4, 52, 52, 29, 4, 0, 29, 45, 23, 4, 52, 52, 4, 45, 6, 0, 4, 29, 52, 52, 45, 4, 6, 12, 4, 0, 52, 52, 27, 6, 29, 23, 52, 45, 4, 0, 3, 29, 27, 41, 7, 52, 4, 0, 27, 41, 6, 29, 52, 4, 23, 7, 27, 23, 6, 29, 52, 45, 34, 0, 27, 6, 41, 29, 52, 23, 4, 0, 30, 29, 27, 6, 41, 52, 4, 0, 45, 27, 7, 29, 6, 52, 4, 0, 29, 27, 41, 6, 7, 52, 34, 0, 17, 7, 27, 29, 41, 52, 4, 23, 7, 52, 52, 41, 4, 19, 13, 6, 12, 13, 12, 29, 13, 7, 7, 29, 52, 52, 45, 23, 0, 12, 13, 12, 7, 29, 52, 52, 45, 23, 0, 6, 13, 12, 13, 12, 7, 7, 52, 52, 45, 23, 0, 13, 12, 13, 12, 29, 7, 13, 52, 52, 45, 23, 0, 6, 12, 13, 29, 12, 29, 6, 12, 7, 45, 52, 52, 23, 0, 13, 12, 13, 7, 29, 52, 52, 41, 4, 0, 13, 6, 12, 13, 12, 29, 6, 12, 7, 52, 52, 45, 13, 4, 0, 12, 13, 7, 29, 12, 52, 52, 41, 4, 0, 29, 12, 13, 13, 41, 6, 52, 1, 52, 52, 4, 4, 23, 52, 1, 52, 6, 52, 4, 44, 52, 52, 4, 0, 29, 52, 6, 52, 4, 44, 52, 52, 4, 0, 1, 52, 52, 6, 4, 52, 52, 44, 4, 0, 52, 1, 52, 52, 6, 4, 44, 52, 4, 0, 52, 1, 43, 41, 52, 52, 4, 52, 4, 20, 29, 52, 41, 52, 52, 11, 4, 52, 4, 0, 29, 52, 52, 6, 10, 52, 44, 52, 4, 0, 45, 52, 1, 4, 52, 52, 6, 52, 4, 0, 45, 29, 43, 52, 12, 4, 52, 52, 4, 0, 41, 5, 23, 3, 29, 45, 52, 11, 4, 52, 2, 5, 3, 2, 5, 45, 0, 5, 4, 45, 23, 52, 11, 29, 3, 5, 41, 0, 41, 23, 11, 52, 2, 5, 29, 4, 5, 2, 11, 0, 45, 5, 3, 4, 41, 23, 52, 29, 23, 4, 45, 2, 5, 5, 3, 39, 41, 11, 23, 52, 2, 3, 45, 29, 5, 5, 23, 41, 23, 52, 11, 4, 41, 29, 3, 5, 45, 23, 11, 4, 23, 52, 2, 5, 5, 29, 5, 23, 41, 3, 5, 23, 11, 4, 23, 52, 5, 23, 5, 29, 2, 37, 3, 41, 23, 11, 4, 52, 5, 5, 2, 3, 11, 41, 41, 23, 4, 23, 52, 29, 23, 52, 41, 41, 23, 16, 10, 3, 3, 29, 11, 52, 23, 16, 29, 4, 2, 5, 11, 41, 6, 52, 45, 52, 4, 23, 23, 52, 5, 41, 11, 29, 3, 16, 52, 45, 23, 3, 5, 45, 16, 23, 41, 29, 11, 4, 23, 52, 11, 29, 10, 5, 23, 41, 23, 45, 3, 52, 16, 52, 23, 41, 4, 29, 5, 11, 2, 16, 3, 52, 45, 52, 29, 23, 5, 11, 10, 16, 23, 3, 41, 23, 45, 52, 23, 41, 0, 3, 16, 29, 5, 23, 11, 23, 45, 52, 11, 23, 3, 23, 41, 16, 29, 5, 41, 52, 4, 52, 29, 23, 41, 23, 11, 52, 16, 10, 3, 3, 41, 23, 3, 27, 16, 11, 2, 5, 5, 52, 52, 52, 52, 29, 4, 21, 3, 5, 2, 5, 5, 52, 27, 11, 52, 52, 52, 29, 4, 0, 3, 27, 16, 5, 52, 52, 52, 37, 52, 29, 11, 2, 4, 21, 3, 5, 2, 31, 52, 16, 52, 52, 52, 11, 29, 45, 4, 0, 5, 3, 52, 16, 27, 35, 2, 52, 52, 52, 11, 4, 23, 29, 3, 5, 27, 11, 5, 5, 52, 52, 52, 52, 29, 4, 0, 2, 52, 5, 2, 12, 16, 11, 52, 5, 3, 52, 52, 29, 4, 0, 5, 11, 39, 9, 5, 5, 16, 18, 52, 52, 52, 12, 4, 0, 2, 5, 16, 3, 5, 11, 25, 52, 52, 52, 45, 52, 4, 0, 16, 27, 11, 2, 52, 3, 5, 5, 25, 52, 52, 52, 4, 0, 3, 26, 44, 0, 29, 13, 5, 5, 52, 52, 52, 11, 2, 16, 5, 4, 3, 0, 44, 5, 23, 52, 52, 11, 52, 2, 16, 29, 44, 0, 5, 13, 5, 3, 23, 52, 52, 11, 52, 2, 5, 29, 3, 5, 4, 5, 0, 44, 24, 52, 52, 52, 11, 52, 2, 5, 23, 4, 0, 3, 44, 5, 52, 52, 52, 11, 52, 2, 3, 24, 3, 4, 5, 0, 44, 5, 52, 52, 52, 11, 52, 23, 5, 29, 0, 5, 3, 5, 4, 44, 23, 52, 52, 11, 52, 2, 5, 25, 44, 5, 3, 5, 40, 4, 0, 23, 23, 52, 11, 52, 2, 5, 4, 44, 3, 0, 24, 5, 5, 23, 52, 52, 11, 52, 2, 5, 4, 44, 3, 5, 5, 47, 0, 23, 52, 52, 11, 52, 2, 5, 6, 45, 11, 4, 23, 23, 1, 4, 11, 28, 5, 52, 11, 1, 45, 5, 11, 23, 6, 28, 4, 23, 52, 4, 1, 45, 28, 11, 23, 0, 11, 4, 6, 5, 4, 23, 4, 0, 11, 28, 11, 23, 45, 6, 1, 5, 23, 4, 4, 23, 4, 28, 23, 45, 11, 11, 6, 1, 5, 23, 11, 23, 23, 11, 28, 4, 6, 45, 1, 5, 23, 4, 4, 4, 28, 1, 23, 6, 11, 23, 5, 45, 23, 11, 16, 11, 2, 45, 11, 12, 25, 52, 52, 24, 52, 4, 0, 4, 12, 2, 11, 16, 25, 41, 25, 11, 23, 52, 52, 4, 0, 10, 16, 10, 23, 41, 11, 12, 2, 11, 6, 23, 52, 52, 4, 0, 41, 16, 6, 11, 25, 4, 11, 2, 52, 52, 52, 29, 4, 0, 6, 11, 2, 12, 11, 16, 45, 52, 52, 4, 24, 52, 4, 0, 11, 11, 12, 4, 25, 2, 53, 16, 23, 52, 52, 4, 0, 41, 4, 16, 25, 41, 11, 25, 11, 2, 23, 52, 52, 29, 23, 0, 25, 2, 12, 16, 11, 41, 11, 53, 52, 52, 52, 4, 4, 0, 6, 11, 11, 16, 4, 12, 25, 45, 2, 52, 23, 23, 4, 0, 16, 4, 25, 25, 11, 41, 11, 12, 2, 23, 52, 23, 4, 0, 45, 5, 28, 4, 29, 2, 11, 52, 52, 52, 6, 4, 23, 11, 16, 45, 11, 4, 2, 28, 29, 52, 52, 52, 6, 4, 23, 11, 44, 2, 28, 11, 4, 5, 29, 52, 52, 52, 6, 4, 43, 11, 4, 35, 6, 28, 5, 11, 29, 2, 52, 52, 23, 11, 4, 20, 4, 45, 16, 29, 2, 28, 52, 52, 52, 6, 11, 11, 4, 43, 36, 4, 29, 28, 16, 41, 2, 52, 52, 23, 6, 11, 4, 0, 28, 45, 5, 4, 11, 2, 25, 52, 52, 52, 11, 4, 20, 6, 29, 11, 2, 4, 45, 5, 28, 52, 52, 52, 11, 6, 4, 0, 4, 28, 16, 35, 29, 2, 6, 11, 52, 52, 52, 11, 4, 43, 45, 11, 6, 2, 4, 29, 28, 52, 52, 52, 16, 11, 4, 43, 35, 11, 23, 5, 34, 11, 23, 52, 52, 28, 29, 52, 4, 0, 5, 11, 23, 6, 11, 34, 45, 23, 52, 52, 28, 29, 52, 23, 23, 11, 11, 6, 5, 45, 0, 34, 52, 52, 52, 28, 29, 4, 5, 35, 11, 0, 13, 11, 23, 23, 52, 52, 52, 28, 29, 23, 11, 6, 23, 23, 18, 0, 11, 5, 34, 52, 52, 28, 52, 29, 11, 23, 5, 45, 34, 23, 11, 52, 52, 52, 28, 29, 4, 0, 11, 23, 34, 11, 23, 5, 18, 23, 23, 52, 52, 28, 52, 29, 23, 6, 23, 11, 45, 5, 11, 34, 52, 52, 52, 28, 29, 4, 7, 11, 2, 23, 23, 11, 5, 13, 23, 52, 28, 12, 52, 34, 5, 0, 11, 23, 34, 7, 23, 11, 23, 52, 28, 29, 52, 23, 52, 5, 23, 5, 41, 28, 23, 23, 41, 5, 23, 52, 5, 0, 28, 41, 23, 41, 23, 23, 41, 0, 52, 5, 5, 41, 28, 41, 23, 18, 23, 5, 5, 24, 52, 23, 41, 5, 0, 5, 28, 41, 52, 8, 52, 0, 8, 5, 41, 5, 41, 28, 52, 52, 41, 5, 18, 28, 5, 52, 23, 0, 52, 28, 0, 52, 5, 5, 41, 3, 8, 52, 5, 8, 0, 5, 28, 52, 3, 23, 41, 41, 5, 28, 0, 52, 23, 5, 41, 52, 5, 24, 2, 5, 52, 41, 52, 8, 0, 2, 24, 45, 5, 5, 52, 52, 8, 0, 24, 45, 5, 5, 41, 52, 52, 6, 0, 24, 5, 45, 2, 5, 52, 52, 8, 23, 37, 24, 5, 5, 2, 52, 52, 8, 23, 41, 5, 5, 24, 45, 52, 52, 8, 23, 5, 5, 24, 41, 52, 45, 52, 8, 0, 5, 2, 24, 5, 52, 45, 52, 8, 0, 5, 45, 44, 5, 24, 52, 52, 8, 0, 5, 28, 23, 45, 16, 52, 26, 52, 23, 8, 23, 41, 23, 16, 28, 5, 23, 23, 52, 52, 8, 43, 52, 23, 28, 16, 11, 23, 8, 23, 52, 52, 16, 31, 28, 3, 11, 52, 52, 27, 16, 23, 8, 23, 23, 5, 28, 41, 52, 8, 23, 52, 52, 16, 26, 5, 23, 28, 41, 52, 26, 8, 23, 52, 16, 52, 45, 23, 3, 28, 16, 23, 8, 23, 52, 52, 26, 3, 23, 11, 28, 23, 26, 52, 52, 16, 8, 23, 11, 23, 28, 16, 3, 23, 8, 23, 26, 52, 52, 28, 41, 23, 5, 52, 27, 8, 23, 23, 52, 16, 23, 5, 28, 5, 27, 12, 52, 8, 0, 52, 23, 0, 52, 12, 28, 3, 5, 52, 8, 0, 52, 52, 5, 27, 12, 5, 28, 52, 52, 23, 23, 52, 52, 5, 28, 12, 5, 27, 52, 52, 8, 23, 52, 52, 28, 5, 34, 5, 27, 52, 52, 8, 23, 52, 52, 28, 27, 5, 5, 23, 23, 23, 8, 0, 52, 23, 28, 16, 5, 52, 12, 27, 52, 8, 0, 52, 52, 23, 5, 45, 28, 5, 52, 52, 8, 0, 52, 52, 12, 16, 27, 28, 5, 23, 52, 8, 0, 52, 23, 16, 27, 12, 5, 28, 52, 23, 8, 0, 52, 52, 6, 28, 23, 23, 3, 10, 1, 52, 45, 41, 52, 13, 1, 28, 41, 3, 10, 23, 23, 6, 23, 45, 52, 27, 3, 1, 23, 6, 28, 10, 45, 23, 52, 41, 52, 7, 23, 23, 1, 41, 3, 10, 45, 28, 52, 52, 27, 6, 41, 23, 1, 23, 10, 6, 28, 52, 45, 52, 27, 3, 3, 6, 23, 25, 1, 10, 23, 28, 52, 45, 52, 27, 23, 28, 3, 23, 10, 1, 6, 52, 45, 41, 52, 27, 6, 7, 23, 28, 1, 23, 3, 52, 45, 41, 52, 27, 7, 11, 28, 23, 3, 6, 23, 1, 45, 52, 52, 27, 24, 45, 23, 6, 1, 30, 23, 52, 41, 52, 27, 3, 3, 28, 23, 4, 27, 45, 23, 6, 52, 52, 23, 52, 23, 45, 6, 0, 23, 45, 3, 45, 4, 27, 28, 52, 52, 52, 6, 52, 0, 23, 10, 18, 23, 6, 27, 45, 3, 53, 52, 52, 52, 52, 45, 0, 45, 4, 3, 23, 23, 6, 27, 52, 52, 6, 52, 52, 41, 10, 45, 23, 28, 6, 0, 23, 3, 27, 52, 23, 52, 52, 4, 45, 27, 6, 23, 23, 27, 0, 3, 52, 23, 23, 52, 52, 4, 3, 6, 41, 28, 27, 0, 23, 45, 6, 52, 52, 52, 52, 0, 52, 3, 23, 42, 4, 27, 6, 45, 28, 23, 52, 52, 52, 6, 18, 43, 3, 23, 27, 4, 0, 28, 52, 52, 52, 44, 52, 3, 6, 30, 18, 0, 23, 23, 18, 52, 52, 28, 52, 52, 27, 23, 45, 26, 28, 52, 52, 52, 52, 27, 3, 0, 6, 0, 4, 23, 28, 6, 4, 45, 52, 52, 52, 52, 26, 3, 20, 6, 0, 45, 41, 28, 6, 23, 52, 52, 52, 52, 27, 3, 0, 23, 0, 45, 26, 23, 28, 6, 52, 52, 4, 23, 52, 3, 0, 23, 23, 4, 28, 33, 11, 52, 52, 23, 52, 26, 23, 20, 23, 23, 45, 45, 28, 3, 41, 0, 52, 52, 6, 52, 52, 26, 6, 0, 4, 21, 28, 41, 6, 52, 52, 4, 52, 52, 27, 3, 0, 23, 0, 6, 23, 45, 52, 52, 28, 52, 52, 26, 3, 0, 23, 0, 4, 45, 23, 28, 52, 52, 6, 4, 52, 52, 26, 3, 0, 6, 0, 4, 6, 44, 34, 28, 52, 52, 52, 13, 27, 3, 0, 23, 23, 0, 27, 4, 34, 3, 43, 28, 52, 52, 6, 13, 6, 0, 52, 3, 27, 52, 34, 27, 4, 28, 0, 27, 52, 52, 6, 6, 0, 43, 43, 28, 3, 23, 27, 0, 4, 52, 52, 52, 12, 6, 0, 4, 28, 45, 23, 3, 27, 0, 52, 52, 52, 52, 44, 23, 23, 34, 0, 28, 3, 27, 4, 52, 52, 52, 6, 13, 26, 6, 0, 27, 27, 3, 34, 4, 23, 0, 52, 52, 6, 23, 6, 0, 52, 27, 23, 4, 35, 0, 28, 44, 52, 52, 12, 3, 6, 0, 52, 23, 27, 28, 4, 3, 0, 34, 52, 52, 52, 44, 23, 6, 0, 27, 52, 28, 4, 0, 44, 3, 52, 52, 6, 12, 45, 6, 0, 11, 23, 1, 35, 23, 41, 23, 23, 52, 52, 52, 23, 0, 2, 17, 23, 11, 45, 23, 45, 52, 52, 6, 27, 3, 23, 1, 52, 2, 17, 11, 6, 23, 23, 45, 45, 52, 52, 52, 3, 0, 27, 1, 2, 0, 45, 5, 41, 11, 52, 52, 6, 6, 52, 23, 3, 0, 1, 2, 17, 1, 6, 27, 11, 6, 45, 52, 52, 6, 3, 0, 52, 23, 2, 0, 23, 23, 45, 45, 11, 52, 52, 52, 23, 3, 0, 23, 1, 2, 17, 45, 6, 11, 23, 45, 23, 1, 52, 52, 52, 13, 3, 0, 2, 0, 45, 23, 23, 6, 27, 52, 52, 45, 11, 23, 3, 0, 1, 23, 0, 27, 45, 6, 23, 23, 11, 37, 52, 52, 23, 3, 23, 1, 2, 17, 6, 6, 23, 52, 52, 44, 45, 52, 52, 11, 23, 0, 1, 2, 0, 29, 23, 1, 5, 37, 52, 6, 52, 3, 0, 2, 17, 11, 11, 6, 1, 6, 28, 29, 37, 52, 52, 3, 0, 2, 0, 23, 28, 11, 1, 6, 29, 52, 37, 52, 3, 0, 2, 0, 29, 23, 45, 37, 6, 52, 52, 23, 0, 11, 1, 2, 0, 33, 28, 1, 6, 29, 6, 52, 52, 3, 0, 11, 2, 0, 23, 33, 45, 29, 1, 6, 52, 52, 3, 0, 11, 2, 0, 29, 11, 5, 23, 1, 37, 52, 52, 3, 0, 6, 2, 0, 1, 29, 28, 23, 6, 37, 52, 52, 3, 0, 11, 2, 23, 28, 29, 11, 1, 23, 37, 6, 52, 52, 3, 0, 2, 0, 1, 41, 6, 3, 41, 52, 29, 0, 23, 23, 52, 52, 53, 2, 23, 29, 1, 41, 3, 0, 45, 6, 52, 52, 6, 52, 52, 11, 2, 23, 1, 3, 6, 29, 18, 0, 11, 52, 52, 6, 45, 52, 52, 2, 23, 28, 0, 3, 1, 1, 6, 29, 41, 52, 52, 6, 52, 52, 2, 23, 41, 3, 1, 11, 45, 0, 29, 6, 52, 52, 52, 52, 2, 23, 6, 41, 3, 0, 29, 1, 23, 6, 11, 52, 52, 45, 52, 52, 2, 23, 1, 6, 3, 29, 41, 0, 52, 52, 6, 45, 52, 52, 11, 2, 0, 29, 0, 3, 1, 6, 41, 28, 23, 52, 52, 52, 52, 11, 23, 23, 41, 28, 6, 1, 0, 29, 3, 52, 52, 52, 52, 2, 23, 53, 23, 4, 18, 2, 5, 45, 13, 13, 41, 0, 11, 0, 11, 52, 1, 0, 2, 9, 23, 7, 11, 7, 41, 11, 52, 41, 1, 10, 5, 13, 23, 11, 18, 0, 4, 5, 41, 2, 11, 23, 4, 45, 1, 11, 7, 9, 13, 0, 11, 41, 5, 0, 2, 11, 52, 41, 1, 2, 11, 7, 11, 0, 41, 13, 0, 9, 45, 52, 4, 16, 1, 41, 2, 13, 11, 10, 9, 5, 11, 0, 0, 52, 41, 41, 1, 10, 5, 11, 2, 41, 0, 11, 7, 13, 0, 4, 52, 41, 1, 0, 2, 7, 10, 11, 5, 41, 11, 45, 0, 9, 13, 1, 52, 5, 11, 0, 2, 41, 4, 13, 10, 45, 0, 7, 11, 52, 1, 44, 7, 2, 23, 0, 45, 16, 23, 23, 1, 41, 52, 55, 23, 23, 1, 45, 4, 23, 16, 41, 2, 4, 0, 41, 52, 11, 23, 0, 1, 34, 2, 7, 23, 44, 30, 16, 23, 41, 52, 11, 23, 23, 27, 1, 4, 13, 25, 52, 2, 27, 27, 5, 52, 11, 11, 52, 52, 0, 23, 23, 13, 2, 52, 5, 25, 27, 27, 1, 4, 52, 52, 52, 11, 11, 0, 23, 27, 30, 1, 13, 52, 4, 27, 52, 52, 52, 27, 27, 2, 25, 5, 11, 0, 23, 11, 30, 34, 1, 25, 52, 27, 5, 13, 52, 23, 52, 27, 27, 11, 0, 0, 11, 2, 30, 27, 4, 52, 13, 53, 27, 52, 52, 52, 27, 2, 1, 5, 11, 0, 0, 11, 27, 52, 1, 25, 30, 13, 27, 4, 23, 27, 52, 23, 2, 5, 11, 23, 23, 11, 10, 1, 5, 30, 2, 25, 52, 13, 27, 27, 52, 23, 23, 11, 0, 0, 11, 27, 5, 34, 52, 30, 11, 13, 27, 1, 27, 2, 1, 27, 52, 52, 23, 23, 23, 11, 1, 5, 52, 27, 25, 4, 27, 13, 23, 52, 23, 52, 27, 2, 11, 0, 23, 11, 17, 1, 13, 41, 12, 23, 13, 30, 4, 41, 52, 23, 5, 41, 52, 1, 23, 23, 4, 4, 23, 4, 41, 23, 13, 31, 5, 16, 12, 41, 23, 52, 52, 11, 1, 0, 5, 41, 10, 12, 41, 23, 13, 30, 53, 4, 41, 23, 52, 52, 11, 11, 23, 23, 31, 4, 2, 4, 12, 4, 5, 23, 23, 41, 41, 52, 23, 52, 11, 1, 23, 23, 4, 41, 30, 12, 5, 41, 23, 4, 13, 53, 52, 23, 52, 11, 11, 0, 23, 41, 5, 4, 4, 31, 23, 13, 12, 30, 41, 52, 23, 52, 41, 11, 11, 1, 0, 23, 4, 30, 5, 41, 23, 2, 42, 13, 4, 1, 23, 52, 52, 11, 45, 11, 23, 23, 13, 5, 23, 4, 12, 41, 10, 31, 41, 1, 30, 23, 52, 52, 11, 11, 23, 23, 4, 12, 41, 41, 23, 5, 13, 30, 16, 17, 23, 52, 52, 41, 11, 11, 1, 23, 1, 5, 31, 23, 13, 41, 4, 10, 41, 4, 23, 23, 52, 52, 11, 11, 23, 23, 52, 3, 5, 4, 8, 29, 13, 28, 52, 11, 7, 23, 8, 3, 28, 5, 4, 11, 52, 12, 41, 52, 7, 23, 12, 52, 3, 4, 8, 11, 5, 12, 28, 23, 7, 23, 52, 8, 13, 4, 28, 23, 29, 5, 11, 23, 7, 0, 3, 52, 11, 28, 4, 12, 5, 27, 8, 23, 7, 23, 52, 5, 12, 4, 27, 3, 8, 13, 23, 11, 7, 0, 4, 27, 12, 8, 5, 52, 23, 28, 52, 11, 7, 0, 52, 3, 13, 4, 11, 8, 5, 28, 12, 52, 7, 23, 28, 52, 4, 13, 8, 3, 12, 5, 52, 11, 7, 23, 29, 11, 3, 5, 27, 8, 52, 4, 13, 52, 7, 23, 4, 8, 52, 41, 28, 52, 4, 16, 23, 11, 7, 0, 52, 3, 52, 28, 9, 26, 3, 5, 8, 4, 52, 52, 7, 0, 11, 52, 52, 5, 28, 52, 8, 52, 39, 3, 7, 0, 11, 52, 4, 9, 8, 52, 10, 52, 5, 23, 52, 7, 0, 11, 52, 4, 4, 3, 52, 52, 7, 5, 28, 8, 3, 52, 38, 7, 0, 11, 52, 4, 52, 52, 3, 5, 28, 8, 7, 52, 7, 0, 11, 39, 52, 4, 52, 52, 3, 10, 5, 8, 23, 52, 7, 0, 11, 52, 4, 7, 3, 8, 52, 52, 34, 5, 28, 52, 7, 0, 39, 11, 52, 9, 52, 52, 28, 8, 5, 52, 3, 38, 7, 0, 11, 52, 4, 9, 3, 8, 23, 5, 29, 28, 52, 45, 52, 52, 11, 7, 23, 52, 23, 28, 3, 8, 5, 52, 52, 52, 45, 11, 7, 0, 52, 23, 8, 5, 28, 23, 3, 29, 52, 52, 52, 11, 7, 0, 45, 52, 5, 3, 10, 28, 23, 8, 52, 23, 52, 11, 7, 0, 52, 41, 23, 3, 28, 8, 5, 52, 52, 45, 52, 11, 7, 0, 52, 29, 3, 23, 5, 25, 29, 8, 52, 52, 28, 52, 11, 7, 23, 52, 3, 29, 23, 41, 5, 28, 8, 52, 52, 52, 11, 7, 0, 52, 28, 5, 3, 23, 29, 8, 24, 52, 52, 52, 11, 7, 0, 52, 28, 23, 29, 24, 3, 5, 8, 52, 52, 52, 11, 7, 0, 52, 23, 24, 28, 29, 3, 5, 8, 52, 52, 52, 11, 7, 0, 52, 28, 23, 3, 11, 52, 52, 23, 45, 52, 4, 5, 29, 7, 0, 11, 23, 13, 3, 52, 52, 52, 41, 52, 4, 5, 29, 7, 0, 44, 28, 3, 11, 52, 52, 52, 41, 52, 4, 5, 7, 0, 29, 23, 30, 3, 26, 28, 52, 52, 52, 45, 11, 4, 5, 7, 0, 28, 3, 7, 8, 30, 52, 52, 52, 45, 11, 52, 4, 5, 0, 7, 29, 41, 28, 3, 8, 52, 52, 52, 11, 52, 4, 5, 0, 30, 3, 8, 7, 52, 52, 28, 52, 45, 11, 52, 4, 5, 0, 7, 3, 28, 11, 41, 52, 52, 52, 41, 52, 4, 5, 0, 29, 11, 44, 27, 30, 28, 7, 3, 52, 52, 52, 52, 4, 5, 0, 11, 7, 3, 28, 30, 44, 52, 52, 52, 45, 52, 4, 5, 0, 52, 23, 52, 41, 2, 52, 45, 5, 52, 5, 52, 52, 11, 2, 0, 11, 43, 2, 5, 52, 45, 0, 52, 52, 5, 11, 52, 45, 52, 11, 52, 2, 0, 2, 23, 52, 52, 23, 5, 2, 11, 5, 41, 41, 52, 23, 52, 52, 11, 2, 23, 23, 0, 2, 23, 45, 52, 11, 5, 52, 45, 52, 5, 52, 52, 52, 11, 2, 0, 2, 43, 2, 52, 0, 5, 21, 5, 52, 23, 11, 52, 11, 45, 52, 52, 2, 0, 2, 43, 45, 52, 5, 52, 41, 0, 5, 2, 11, 52, 52, 52, 52, 11, 2, 0, 2, 23, 52, 5, 0, 52, 5, 52, 23, 52, 41, 11, 41, 52, 52, 11, 2, 0, 2, 43, 52, 5, 0, 45, 5, 45, 52, 2, 52, 52, 52, 23, 11, 11, 2, 0, 23, 0, 52, 45, 2, 52, 5, 0, 52, 11, 5, 45, 52, 23, 52, 11, 2, 0, 2, 43, 23, 5, 35, 11, 11, 13, 5, 52, 23, 52, 2, 23, 2, 0, 2, 23, 45, 52, 11, 5, 11, 35, 52, 52, 52, 52, 5, 23, 52, 45, 2, 23, 2, 0, 2, 23, 5, 52, 11, 11, 52, 35, 52, 5, 13, 52, 52, 2, 23, 2, 23, 2, 43, 43, 52, 34, 52, 5, 11, 52, 11, 5, 52, 52, 52, 35, 2, 23, 2, 0, 23, 23, 52, 52, 5, 35, 52, 11, 45, 5, 11, 52, 52, 52, 23, 23, 2, 0, 23, 23, 52, 5, 11, 52, 52, 11, 5, 35, 52, 52, 52, 23, 23, 2, 0, 2, 23, 45, 27, 11, 52, 5, 11, 27, 27, 2, 0, 23, 0, 23, 23, 5, 27, 52, 53, 5, 5, 11, 13, 27, 2, 23, 23, 0, 2, 23, 52, 5, 11, 27, 53, 3, 27, 27, 2, 23, 2, 0, 23, 23]