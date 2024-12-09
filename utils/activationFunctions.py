import numpy as np


"""
sigmoid activation function 
ranges 0 < sigmoid(z) < 1
"""
def sigmoid(z):
    return 1/(1+np.exp(z))


"""
tanh activation function 
ranges -1 <= tanh(z) <= 1
"""
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)-np.exp(-z))

""""
relu activation function (Rectified Linear Unit)
ranges 0 <= relu(z) <= z
"""
def relu(z):
    return max(0, z)

"""
leakyRelu activation function
ranges -inf < leakyRelu(z) < inf
"""
def leakyRelu(z, alpha=0.01):
    if(z > 0):
        return z
    else:
        return alpha*z


"""
elu activation function (Exponential Linear Unit)
ranges 0 <= elu(z) <= z
"""
def elu(z, alpha):
    if(z > 0):
        return z
    else:
        return alpha*(np.exp(z) - 1)


"""
swish activation function
Combines linearity and non-linearity used in some SOTA models
"""
def swish(z):
    return z * sigmoid(z)


"""
mish activation function
Smoother non-linearity; promising in performance.
"""
def mish(z):
    return z*tanh(np.log(1+np.exp(z)))


"""
softplus activation function
Smooth approximation of ReLU.
"""
def softplus(z):
    return np.log(1+np.exp(z))



