# -*- coding: utf-8 -*-
# @Author: nemo-tj
# @Date:   2021-06-06 14:50:47
# @Last Modified by:   nemo-tj
# @Last Modified time: 2021-06-06 14:55:16

from header import *

class Tanh(Activation):
  """docstring for Tanh"""
  def __init__(self, x):
    super(Tanh, self).__init__(x)

  def forward(self):
    self.p = (np.exp(x) - np.exp(np.negative(x))) / (np.exp(x) + np.exp(np.negative(self.x)))
    return self.p

  def backward(self):
    self.derivative = 1. - np.power(self.p, 2)
    return self.derivative


if __name__ == '__main__':
  x = np.linspace(-10, 10, 500)
  plt.plot(x, Tanh(x)()[0], label ='Tanh forward')
  plt.plot(x, Tanh(x)()[1], label ='Tanh backward')
  plt.legend(loc='best')
  plt.show()