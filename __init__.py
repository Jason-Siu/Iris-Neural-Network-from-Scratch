from sklearn import datasets # import dataset
import numpy as np

# seed randomness
np.random.seed(15)

# create target input vector
# this could've been done more elegantly, but regardless it still works
def inputn():
    inputt = []
    for i in range(30):
        inputt.append(x[i])
        inputt.append(x[i + 50])
        inputt.append(x[i + 100])
    return inputt

# defines target vector
# again, could've been done more elegantly
def target():
    target = []
    for i in range(30):
        target.append(y[i])
        target.append(y[i + 50])
        target.append(y[i + 100])
    return target

# initialize weights
def initialize_weights(x,y):
    weight = []
    for i in range(x):
        inner = []
        for j in range(y):
            inner.append(np.random.randn())
        weight.append(inner)
    return weight

# returns bias for the layers
def getBiases(size):
    array = []
    bias = np.random.randn()
    for i in range(size):
        array.append(bias)
    return array

# function that returns the target matrix based on classification
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

# returns vector form of classification
def whatis(x):
    if x == 0:
        return [1,0,0]
    elif x == 1:
        return [0,1,0]
    else:
        return [0,0,1]

# sigmoid activation function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

# finds the max index to make guess
def makeguess(z):
    index = 0 if z[0] > z[1] else 1
    second = index if z[index] > z[2] else 2
    if second == 0:
        print("My guess is: " + str(names[0]))
    elif second == 1:
        print("My guess is: " + str(names[1]))
    else:
        print("My guess is: " + str(names[2]))
        
# displays name of class
def returnGuess(z):
    index = 0 if z[0] > z[1] else 1
    second = index if z[index] > z[2] else 2
    if second == 0:
        return names[0]
    elif second == 1:
        return names[1]
    else:
        return names[2]
# load dataset and define variables
iris = datasets.load_iris()
x = iris.data
y = iris.target
practice = iris.target
names = iris.target_names

# learning rate
alpha = .05

#define test input matrix and target vectors
test_data = np.array(inputn())
target = np.array(target())

# initialize weights and biases
weight_ih = np.array(initialize_weights(4, 4))
weight_ho = np.array(initialize_weights(4, 3))
bias_i = getBiases(4)
bias_h = getBiases(3)

# define target matrix
y = getTarget(target)

# initialize biases
first_bias = bias_i
second_bias = bias_h

#initialize weights
syn0 = weight_ih
syn1 = weight_ho



# Training loop
def train(x, syn0, syn1, first_bias, second_bias):
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
    
    second_bias += alpha * l2_error
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    first_bias += alpha * l1_error
    # adjust weights
    syn1 += alpha * l1.T.dot(l2_delta)
    syn0 += alpha * l0.T.dot(l1_delta)


# training loop
for i in range(50000):
    train(i, syn0, syn1, first_bias, second_bias)
    
go = True
np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

print()

''' 
    if you don't want to train, here are the weight matrices:
    
    syn0 = [[2.2408217095 -1.6853796665 5.5913811924 -0.9366177497]
         [0.6090305537 -2.2210590652 3.3860760754 -0.4248610653]
         [-4.3583983502 2.6291739327 -7.1529550442 -0.1713271684]
         [-2.7078773548 2.4765996026 -4.9313664416 0.6101184294]]
         
    syn1 = [[9.9450263133 -12.7935363317 -5.1024261598]
         [-7.2674701502 -6.2178006814 6.4532717444]
         [-4.4043524385 7.1407092810 -7.7961001105]
         [-0.8959132447 0.7941216092 0.5851137336]]
         
    the weights in this case, we randomly assign very small values,
    and so they don't matter much
    
    but here they are anyways:
    
    first_bias = [0.022111141705110405, 0.022111141705110405, 0.022111141705110405, 0.022111141705110405]
    second bias = [-0.004143583522698942, -0.004143583522698942, -0.004143583522698942]

'''

print("Testing time...")
print()

'''
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
'''

# enter an index: prints predicted output
success = 0
for i in range(150):
    l0 = x[i]
    l1 = nonlin(np.dot(l0,syn0) + first_bias)
    l2 = nonlin(np.dot(l1,syn1) + second_bias)
    
    if(str(returnGuess(l2)) == str(names[int(i / 50)])):
        success += 1
    else:
        print(i)
        
        
print(success/150)
        
    


    
    

    
    
    




    
    




    
    






    
    


