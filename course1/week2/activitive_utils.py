import numpy as np
import math

# sigmoid
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# sigmoid_derivative
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds

# lost function
def lost(A,Y):
    return -(np.dot(Y, np.log(A.T+1e-5)) + np.dot(1 - Y, np.log((1 - A).T+1e-5)))

# total cost function
def cost(A,Y):
    m = Y.shape[1]
    return lost(A,Y)/m

def tom_sigmoid(inx):
    # 对sigmoid函数的优化，避免了出现极大的数据溢出
    if inx >= 0:
        return 1.0/(1+math.exp(-inx))
    else:
        return math.exp(inx)/(1+math.exp(inx))

# opt sigmoid
def opt_sigmoid(x):
    tom_sigmoid_vec = np.vectorize(tom_sigmoid)
    return tom_sigmoid_vec(x)

def round_prob(x):
    return int(round(x))