import numpy as np

#input
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

# actual output
y = np.array([[0,0,1,1]]).T

np.random.seed(1)

#initialize weights with mean 0
w0 = 2*np.random.random((3,1)) - 1

numiter=60000

for i in xrange(numiter):

    # Forward propagation
    a0 = X
    a1 = 1 / (1+ np.exp(-(np.dot(a0, w0))))

    # Backward propagation

    # difference between actual output and predicted outpt
    a1_error = y-a1

    # mulpily the error with the slope of sigmoid
    a1_delta = a1_error*(a1*(1-a1))

    # update weights
    w0 += a0.T.dot(a1_delta)

print "feed forward after training"
print a1
