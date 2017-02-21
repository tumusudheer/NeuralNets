import numpy as np

#input
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

# actual output
y = np.array([[0,1,1,0]]).T

w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1

#number of iterations
numiter = 60000

for i in xrange(numiter):

    # forward propagation
    a0 = 1 / (1+ np.exp(-(np.dot(X, w0))))
    a1 = 1 / (1+np.exp(-(np.dot(a0, w1))))

    # backward propagation
    a1_delta = (y-a1)*(a1*(1-a1))
    a0_delta = a1_delta.dot(w1.T)*(a0*(1-a0))

    # update weights
    w1 += a0.T.dot(a1_delta)
    w0 += X.T.dot(a0_delta)

print "feed forward after training"
print a1
