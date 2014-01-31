import cPickle, gzip, numpy
from numpy import *
import scipy.optimize
import matplotlib.pyplot as plt

#open data, get train, validation and test set
f = gzip.open('mnist.pkl.gz','rb')
train_set,valid_set,test_set = cPickle.load(f)
f.close()

#useful dimensions
train_dim, test_dim = (50000,1), (10000,1)

#Set up training matrix,X, consisting of 50,000 examples of input length 784, add a column of ones for bias
xtrain = hstack((ones(train_dim),train_set[0]))

#Set up column vector,y, of correct output of the 50,000 training examples, y is [50000x1]
ytrain = train_set[1].reshape(train_dim)

#cross-validation set + bias and outcomes
xcv = hstack((ones(test_dim),valid_set[0]))
ycv = valid_set[1].reshape(test_dim)

#test set + bias and outcomes
xtest = hstack((ones(test_dim),test_set[0]))
ytest = test_set[1].reshape(test_dim)

#randomly initialize theta1 [300x785] matrix mapping 485 input terms to 300 units in hidden layer
epsilon1 = sqrt(6)/sqrt(785+300)
theta1 = numpy.random.uniform(-1,1,size=(300,785))*epsilon1

#initialize theta2 [10x301] matrix mapping 301 hidden layer terms to 10 units in output layer
epsilon2 = sqrt(6)/sqrt(300+10)
theta2 = numpy.random.uniform(-1,1,size=(10,301))*epsilon2

#sigmoid function
def sigmoid(x):
    return 1.0/(1.0+numpy.power(e,-x))

#sigmoid gradient function
def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))
    

#feedforward propagation, col dim of X must be 785 (784 input + bias), then the input is transposed
def feedforward(theta_1,theta_2,X):
    a1 = X.transpose()
    z2 = dot(theta_1,a1)
    a2 = sigmoid(z2)
    a2 = vstack((ones(X.shape[0]),a2)) #add bias
    z3 = dot(theta_2,a2)
    a3 = sigmoid(z3) #the hypothesis

    return array(a3) #row dim  = 10, col dim = #training examples

#neural network cost function
def neural_cost(theta_1,theta_2,X,y,reg):
    m = len(y)
            
    #convert numerical y values to 10x1 binary vectors
    ynew = []
    for i in range(m):
        ynew.append((array(range(10)).reshape(10,1)==y[i]).astype(int)) # ynew is list object with m elements of type array [10x1]
        
    cost = 0
    for i in range(m):
        hX = feedforward(theta_1,theta_2,X[i,:].reshape(1,785))
        a = -ynew[i]*log(hX)
        b = (1-ynew[i])*log(1-hX)
        cost = sum(a - b) + cost
        
    cost = cost/m

    reg_cost = sum(theta_1**2) + sum(theta_2**2)
    reg_cost = reg*reg_cost/(2*m)
    
    return cost + reg_cost

#backpropagation algormitms with regularization
def backprop(theta_1,theta_2,X,y,reg):
    m = len(y)

    #convert numerical y values to 10x1 binary vectors
    ynew = []
    for i in range(m):
        ynew.append((array(range(10)).reshape(10,1)==y[i]).astype(int)) # ynew is list object with m elements of type array [10x1]

    #declare error accumulators for theta_1 and theta_2, delta1 and delta2 respectively
    delta1 = zeros((300,785))
    delta2 = zeros((10,301))
    for i in range(m):
        x_i = X[i,:].reshape(1,785)
        a3 = feedforward(theta_1,theta_2,x_i) #hypothesis
        z2 = dot(theta_1,x_i.transpose()) #300x1
        a2=sigmoid(z2) #300x1
        a2=vstack((1,a2)) #301x1
        s3 = a3 - ynew[i] #error associated with output layer, 10x1
        s2 = (dot(theta_2.transpose(),s3)[1:,:])*sigmoid_grad(z2)#300x1, no error associated with bias
        delta2 = delta2 + dot(s3,a2.transpose())
        delta1 = delta1 + dot(s2,x_i)

    theta1_no_bias = hstack((zeros((300,1)),theta_1[:,1:]))
    theta2_no_bias = hstack((zeros((10,1)),theta_2[:,1:]))

    theta1grad = delta1/m + reg*theta1_no_bias/m
    theta2grad = delta2/m + reg*theta2_no_bias/m

    return (theta1grad,theta2grad)
        
def neural_gradient_descent(theta_1,theta_2,X,y,alpha,num_iters,reg):
    for i in range(num_iters):
        neural_grad = backprop(theta_1,theta_2,X,y,reg)
        theta_1 = theta_1 - alpha*neural_grad[0]
        theta_2 = theta_2 - alpha*neural_grad[1]
        

    return (theta_1,theta_2)

def neural_minibatch_gradient_descent(theta_1,theta_2,X,y,alpha,num_iters,reg,b):
    m = len(y)
    alpha_change = (alpha-0.001)/num_iters
    
    traincost=zeros((1,num_iters))
    cvcost=zeros((1,num_iters))
    testcost=zeros((1,num_iters))
    
    for i in range(num_iters):
        
        traincost[0][i]=neural_cost(theta_1,theta_2,xtrain,ytrain,reg)
        cvcost[0][i]=neural_cost(theta_1,theta_2,xcv,ycv,reg)
        testcost[0][i]=neural_cost(theta_1,theta_2,xtest,ytest,reg)
        
        for j in range(m)[0::b]:
            (theta_1,theta_2) = neural_gradient_descent(theta_1,theta_2,X[j:j+b],y[j:j+b],alpha,1,reg)
        alpha = alpha - alpha_change
        
    return (theta_1,theta_2,traincost,cvcost,testcost)

def neural_predict(theta_1,theta_2,X):
    return numpy.argmax(feedforward(theta_1,theta_2,X),axis=0) #nd matrix, where n is the row dimension of X

    

def cost_curve(X_train,y_train,theta_1,theta_2,reg,alpha,num_iters,b):
    cost = zeros((num_iters,1))
    m = len(y_train)
    for i in range(num_iters):
        n = neural_minibatch_gradient_descent(theta_1,theta_2,X_train,y_train,alpha,i,b)
        cost[i] = neural_cost(n[0],n[1],xtrain,ytrain,reg)
    return cost

#visualize hidden layer
def visualize(theta_1):
    for i in range(300):
    im=theta_1[i,1:].reshape(28,28)
    plt.imshow(im,cmap="Greys")
    
                            
    
   
print "Welcome to Exercise 2: Neural Network"
print "Regularization = 0"
print "Initial alpha = 1, scaled down to 0.001"
print "Num Epochs = 20"
print "Batch Size = 50"
print "Please wait approx. 90 MINUTES: training neural network......."
n = neural_minibatch_gradient_descent(theta1,theta2,xtrain,ytrain,1,20,0,50)
print "Done training neural network"
print n[0], n[1]
print "Error rate:",100-sum(neural_predict(n[0],n[1],xtest).reshape(10000,1)==ytest)/100,"%"
print "Cost on training set:",neural_cost(n[0],n[1],xtrain,ytrain,0)
visualize(n[0])
plt.show()