import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------
# Multi-Layer-Perceptron with 2 Layers and K output units for Classification with K classes
# -------------------------------------------------------------------------------------------
def softmax(a): # compute softmax function for potential vector a; for numerical stability
    e_a = np.exp(a - np.max(a)) # subtract maximum potential such that max. exponent is 1
    return e_a / e_a.sum() # return softmax function value
def forwardPropagateActivity(x,W1,W2,b=1.0): # prop. activity x through weights W1,W2 with bias b
    a_1 = np.dot(W1,x); # compute dendritic potentials of hidden layer 1
    z_1 = np.tanh(a_1); # compute activity z_1 of hidden layer 1
    a_2 = np.dot(W2,np.append(z_1,[b])); # compute potentials of output layer 2; extend z1 by b
    z_2 = softmax(a_2); # compute softmax activations of output layer 2
    return z_1, z_2; # return activities in layers 1 and 2; z_2 is y here
def backPropagateErrors(z_1,z_2,t,W1,W2): # backpropagate error signals delta_L
    y=z_2 # layer 2 is output layer
    delta_2=y-t; # Initializing error signals in output layer 2
    alpha_1=np.dot(W2.T,delta_2)[:-1] # error potentials in hidden layer 1 by backprop.; skip bias
    h_prime=(1.0-np.multiply(z_1,z_1)) # (1-z_1.*z_1) is h’(a) for tanh sigmoid function
    delta_1=np.multiply(h_prime,alpha_1) # compute error signals in hidden layer 1
    return delta_1, delta_2 # return error signals for each layer
def doLearningStep(W1,W2,xn,tn,eta,lmbda_by_N,b=1.0): # do one backpropagation learning step...
    z_1 ,z_2 =forwardPropagateActivity(xn,W1,W2,b); # forward propagation of input xn
    delta_1,delta_2=backPropagateErrors(z_1,z_2,tn,W1,W2); # get error signals by backpropagation
    nablaED_1 = np.outer(delta_1,xn) # gradient of data error function for first layer
    nablaED_2 = np.outer(delta_2,np.append(z_1,[b])) # gradient of data error for second layer
    W1=W1*(1.0-lmbda_by_N*eta)-eta*nablaED_1 # update weights of layer 1 with "weight decay" reg.
    W2=W2*(1.0-lmbda_by_N*eta)-eta*nablaED_2 # update weights of layer 2 with "weight decay" reg.
    return W1,W2 # return new weights
def computeError(W1,W2,X,T,b=1.0): # crossentropy error function for data set (X,T) for weights W1,W2
    N,D = X.shape # get size of data set
    E=0; # initialize error with 0
    for n in range(N): # test all data vectors
        y=forwardPropagateActivity(X[n,:].T,W1,W2,b)[1]; # get output values y
    t=T[n,:]; # get actual target vector (should be "one hot" coded)
    e=[-t[i]*np.log(y[i]) for i in range(len(t)) if t[i]>0] # error contributions of y and t
    E=E+np.sum(e) # add sum of component errors to total error
    return E; # return final error value
def plotDecisionSurface(W1,W2,gridX,gridY,dataX1,dataX2,contlevels,epoch,b=1): # plot decision surface (only for K=2 and D=2)
    m,n=gridX.shape
    gridZ=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            yn=forwardPropagateActivity([1,gridX[i,j],gridY[i,j]],W1,W2,b)[1]   # activity for input xn
            gridZ[i,j]=np.log(yn[0]/yn[1])                                                 # plot contours of log-odds-ratio 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.array(dataX1)[:,0],np.array(dataX1)[:,1], c='r', marker='x', s=200)
    ax.scatter(np.array(dataX2)[:,0],np.array(dataX2)[:,1], c='g', marker='*', s=200)
    CS=ax.contour(gridX, gridY, gridZ,levels=contlevels)
    ax.clabel(CS,CS.levels,inline=True)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Log-Odds-Contours after learning epoch '+str(epoch))
    return fig,ax
# *******************************************************
# Main program
# *******************************************************
# (i) Create training data
X1 = np.array([[-2,-1], [-2,2], [-1.5,1], [0,2], [2,1], [3,0], [4,-1], [4,2]]) # class 1 data
N1,D1 = X1.shape
T1 = np.array(N1*[[1.,0]]) # corresponding class labels with one-hot coding: [1,0]=class 1;
X2 = np.array([[-1,-2],[-0.5,-1],[0,0.5],[0.5,-2],[1,0.5],[2,-1],[3,-2]]) # class 2 data
N2,D2 = X2.shape
T2 = np.array(N2*[[0,1.]]) # corresponding class labels with one-hot coding: [0,1]=class 2
X = np.concatenate((X1,X2)) # entire data set
T = np.concatenate((T1,T2)) # entire label set
N,D = X.shape
X=np.concatenate((np.ones((N,1)),X),1) # X is extended by a column vector with ones (bias weights w_j0)
N,D = X.shape # update size parameters
N,K = T.shape # update size parameters
# (ii) Train MLP
M=3 # number of hidden units (without bias unit)
b=1.0 # activity of bias unit in hidden layer (0=no bias unit)
eta=0.01 # learning rate
lmbda=0 # regularization coefficient
nEpochs=500 # number of learning epochs
contlevels=[-1,0,1]                # plot contour levels (of log-odds-ratio)
epochs4plot=[-1,0,5,10,50,100,nEpochs-1] # learning epochs for which a plot will be made
gridX,gridY = np.meshgrid(np.arange(-3,5,0.1),np.arange(-3,3,0.1))  # mesh grid for plot
W1=1.0*(np.random.rand(M,D)-0.5) # initialize weights of layer 1 randomly
W2=1.0*(np.random.rand(K,M+1)-0.5) # initialize weights of layer 2 randomly
E=computeError(W1,W2,X,T,b)
print("initial error E=",E)
if -1 in epochs4plot: plotDecisionSurface(W1,W2,gridX,gridY,X1,X2,contlevels,-1,b)
for epoch in range(nEpochs): # loop over learning epochs
    errc = 0 # initialize classification errors with zero
    for n in range(N): # loop over all training data
        xn=X[n,:] # n-th data vector
        tn=T[n,:] # n-th target value
        yn=forwardPropagateActivity(xn,W1,W2,b)[1] # test training vector xn
        yhat,that=2,2 # initialize class labels
        if(tn[0]>=tn[1]): that=1 # actual class label
        if(yn[0]>=yn[1]): yhat=1 # predicted class by MLP
        if(yhat!=that): errc=errc+1 # count classification error
        W1,W2=doLearningStep(W1,W2,xn,tn,eta,lmbda/N,b) # do one backprop learning update of weights    
    E=computeError(W1,W2,X,T,b)
    print("after epoch ", epoch, " error function E=",E, " and classification errors = ", errc)
    if epoch in epochs4plot: plotDecisionSurface(W1,W2,gridX,gridY,X1,X2,contlevels,epoch,b)
plt.show()