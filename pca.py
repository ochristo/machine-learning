#this script will do a principal component analysis on the MNIST dataset and produce a scatterplot
import cPickle, gzip, numpy
from numpy import *
import matplotlib.pyplot as plt

#open data, get train, validation and test set
f = gzip.open('mnist.pkl.gz','rb')
train_set,valid_set,test_set = cPickle.load(f)
f.close()

#useful dimensions
train_dim, test_dim = (50000,1), (10000,1)

#Set up training matrix,X, consisting of 50,000 examples of input length 784, add a column of ones for bias
xtrain = train_set[0]

#Set up column vector,y, of correct output of the 50,000 training examples, y is [50000x1]
ytrain = train_set[1].reshape(train_dim)

#cross-validation set + bias and outcomes
xcv = valid_set[0]
ycv = valid_set[1].reshape(test_dim)

#test set + bias and outcomes
xtest = test_set[0]
ytest = test_set[1].reshape(test_dim)

print "Reducing dimensions and plotting, please wait....."

covariance = dot(xtrain.transpose(),xtrain) # 784x784 matrix, symmetric positive semi-definite
eigenvectors = (numpy.linalg.eig(covariance))[1] # 784x784 matrix, columns are unit eigenvectors of covariance matrix
projection_vectors = eigenvectors[:,:2] # 784x2 matrix

def dim_reduce(vectors):
    return dot(projection_vectors.transpose(),vectors.transpose())

def scatterplot(points,i,marker):
    return plt.plot(points[i][:1,:],points[i][1:2,:],marker)

#take approx. 1000 test examples of each class, 10000 in total to make scatterplot

t=[]

for i in range(10):
    t.append([])

for i in range(10000):
    t[ytest[i]].append(xtest[i])

for i in range(10):
    t[i] = dim_reduce(asarray(t[i]))

markers = ["yo","mo","co","ro","go","bo","ko","rx","bx","kx"]

for i in range(10):
    scatterplot(t,i,markers[i])


print "Now showing MNIST PCA Scatterplot..."    
plt.show()