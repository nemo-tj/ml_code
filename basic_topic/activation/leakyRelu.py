# -*- coding: utf-8 -*-
# @Author: nemo-tj
# @Date:   2021-06-06 15:06:46
# @Last Modified by:   nemo-tj
# @Last Modified time: 2021-06-06 15:07:44
from header import *

class LeakyRelu(Activation):
    def __init__(self, x, alpha=0.1):
        super(LeakyRelu, self).__init__(x)
        self.alpha = alpha

    def forward(self):
        self.p = np.maximum(self.alpha * self.x, self.x)
        return self.p

    def backward(self):
        self.derivative = np.full_like(self.p, 1)
        self.derivative[self.p < 0] = self.alpha
        return self.derivative

if __name__ == "__main__":
    x = np.linspace(-10, 10, 500)
    plt.plot(x, LeakyRelu(x)()[0], label='leakyRelu_forward')
    plt.plot(x, LeakyRelu(x)()[1], label='leakyRelu_backward')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
