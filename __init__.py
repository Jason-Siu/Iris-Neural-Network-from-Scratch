from sklearn import datasets
import numpy as np


np.random.seed(14)

def inputn():
    inputt = []
    for i in range(10):
        inputt.append(x[i])
        inputt.append(x[i + 50])
        inputt.append(x[i + 100])
    return inputt

def target():
    target = []
    for i in range(10):
        target.append(y[i])
        target.append(y[i + 50])
        target.append(y[i + 100])
    return target
    
def initialize_weights(x,y):
    weight = []
    for i in range(x):
        inner = []
        for j in range(y):
            inner.append(np.random.randn())
        weight.append(inner)
    return weight

def getBiases(size):
    array = []
    bias = np.random.randn()
    for i in range(size):
        array.append(bias)
    return array


def getTarget(x):
    res = []
    for i in range(x.size):
        if x[i] == 0:
            res += [[1,0,0]]
        elif x[i] == 1:
            res += [[0,1,0]]
        else:
            res += [[0,0,1]]
    return np.array(res)

def whatis(x):
    if x == 0:
        return [1,0,0]
    elif x == 1:
        return [0,1,0]
    else:
        return [0,0,1]

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

def makeguess(z):
    index = 0 if z[0] > z[1] else 1
    second = index if z[index] > z[2] else 2
    if second == 0:
        print("My guess is: " + str(names[0]))
    elif second == 1:
        print("My guess is: " + str(names[1]))
    else:
        print("My guess is: " + str(names[2]))
    
iris = datasets.load_iris()
x = iris.data
y = iris.target
practice = iris.target
names = iris.target_names

alpha = .15

test_data = np.array(inputn())
target = np.array(target())
 
weight_ih = np.array(initialize_weights(4, 4))
weight_ho = np.array(initialize_weights(4, 3))
bias_i = getBiases(4)
bias_h = getBiases(3)

y = getTarget(target)

first_bias = bias_i
second_bias = bias_h

syn0 = weight_ih
syn1 = weight_ho

# Training loop
def forward(x, syn0, syn1, first_bias, second_bias):
    # input layer = test_data of 30 samples
    l0 = test_data
    
    #forward pass
    l1 = nonlin(np.dot(l0,syn0) + first_bias)
    l2 = nonlin(np.dot(l1,syn1) + second_bias)
    
    l2_error = y - l2
    if (x % 1000) == 0:
        print()
        print ("Error:" + str(np.mean(np.abs(l2_error))))
    
    # backprop
    l2_delta = l2_error * nonlin(l2, deriv = True)
    
    second_bias += alpha * nonlin(l2, deriv = True)
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    first_bias += alpha * nonlin(l1,deriv=True)
    # adjust weights
    syn1 += alpha * l1.T.dot(l2_delta)
    syn0 += alpha * l0.T.dot(l1_delta)



for i in range(50000):
    forward(i, syn0, syn1, first_bias, second_bias)
    
go = True
np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})
print()
print("Testing time...")
print()

while go:
    print()
    index = int(input("Enter a number (between 0 - 149) : "))
    l0 = x[index]
    l1 = nonlin(np.dot(l0,syn0) + first_bias)
    l2 = nonlin(np.dot(l1,syn1) + second_bias)
    print("My guess (in vector form): " + str(l2))
    makeguess(l2)
    print("Correct Answer: " + str(names[practice[index]]))
    again = input("Go again? [y/n]  ")
    if again == 'n':
        go = False
    
    


    
    

    
    
    




    
    




    
    






    
    


