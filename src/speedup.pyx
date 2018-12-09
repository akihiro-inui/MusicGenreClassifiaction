#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import cython
import numpy as np
cimport numpy as cnp


def func1(int n):
    cdef:
        int i, sum
        list hoge
    hoge = []
    for i in range(n):
        sum += i
        hoge.append(i)
    return sum, hoge

cdef func2():
    pass

cpdef func3():
    pass