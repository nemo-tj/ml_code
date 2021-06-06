# -*- coding: utf-8 -*-
# @Author: nemo-tj
# @Date:   2021-06-06 14:37:31
# @Last Modified by:   nemo-tj
# @Last Modified time: 2021-06-06 15:32:09

import numpy as np
import matplotlib.pyplot as plt

# https://zhuanlan.zhihu.com/p/30385380

class Activation(object):
  """docstring for Activation"""
  def __init__(self, x):
    self.x = x
    self.p = None
    self.derivative = None

  def forward(self):
    pass

  def backward(self):
    pass

  def __call__(self):
    res = list()
    res.append(self.forward())
    res.append(self.backward())
    return res
    