# -*- coding: utf-8 -*-
# @Author: nemo-tj
# @Date:   2021-06-06 15:01:38
# @Last Modified by:   nemo-tj
# @Last Modified time: 2021-06-06 15:33:42

from header import *

class Identity(Activation):

  def __init__(self, x):
    super(Identity, self).__init__(x)

  def forward(self):
    self.p = self.x
    return self.p

  def backward(self):
    self.derivation = np.full_like(self.p, 1)
    return self.derivation

if __name__ == '__main__':
  x = np.linspace(-10, 10, 500)
  plt.plot(x, Identity(x)()[0], label='identity_forward')
  plt.plot(x, Identity(x)()[1], label='identity_backward')
  plt.legend(loc='best')
  plt.grid()
  plt.show()