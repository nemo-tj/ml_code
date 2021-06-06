# -*- coding: utf-8 -*-
# @Author: nemo-tj
# @Date:   2021-06-06 14:43:47
# @Last Modified by:   nemo-tj
# @Last Modified time: 2021-06-06 14:50:06
import numpy as np
import matplotlib.pyplot as plt
from activation import Activation

class Sigmoid(Activation):
  """docstring for Sigmoid"""
  def __init__(self, x):
    super(Sigmoid, self).__init__(x)

  def forward(self):
    self.p = 1. / (1 + np.exp(np.negative(self.x)))
    return self.p

  def backward(self):
    self.derivative = self.p * (1 - self.p)
    return self.derivative


if __name__ == '__main__':
  x = np.linspace(-10, 10, 500)
  plt.plot(x, Sigmoid(x)()[0], label ='Sigmoid forward')
  plt.plot(x, Sigmoid(x)()[1], label ='Sigmoid backward')
  plt.legend(loc='best')
  plt.show()