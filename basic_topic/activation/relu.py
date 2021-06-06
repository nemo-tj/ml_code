# -*- coding: utf-8 -*-
# @Author: nemo-tj
# @Date:   2021-06-06 14:55:41
# @Last Modified by:   nemo-tj
# @Last Modified time: 2021-06-06 15:00:52

from header import *

class ReLu(Activation):
  """docstring for ReLu"""
  def __init__(self, arg):
    super(ReLu, self).__init__(arg)

  def forward(self):
    self.p = np.maximum(0., self.x)
    return self.p

  def backward(self):
    self.derivation = np.sign(self.p)
    return self.derivation


if __name__ == "__main__":
    x = np.linspace(-10, 10, 500)
    plt.plot(x, ReLu(x)()[0], label='relu_forward')
    plt.plot(x, ReLu(x)()[1], label='relu_backward')
    plt.legend(loc='best')
    plt.show()