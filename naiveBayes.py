import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math

#Getting Data
iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features.
y = iris.target

#Splitting into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)

#Calculating mean for each feature of each class
sum0_feature1 = 0
sum0_feature2 = 0
sum0_feature3 = 0
sum0_feature4 = 0


sum1_feature1 = 0
sum1_feature2 = 0
sum1_feature3 = 0
sum1_feature4 = 0

sum2_feature1 = 0
sum2_feature2 = 0
sum2_feature3 = 0
sum2_feature4 = 0

count0 = 0
count1 = 0
count2 = 0
for i in range(0,y_train.size):
    if(y_train[i] == 0):
        sum0_feature1 = sum0_feature1 + X_train[i][0]
        sum0_feature2 = sum0_feature2 + X_train[i][1]
        sum0_feature3 = sum0_feature3 + X_train[i][2]
        sum0_feature4 = sum0_feature4 + X_train[i][3]
        count0 = count0 + 1
    elif(y_train[i] == 1):
        sum1_feature1 = sum1_feature1 + X_train[i][0]
        sum1_feature2 = sum1_feature2 + X_train[i][1]
        sum1_feature3 = sum1_feature3 + X_train[i][2]
        sum1_feature4 = sum1_feature4 + X_train[i][3]
        count1 = count1 + 1
    elif(y_train[i] == 2):
        sum2_feature1 = sum2_feature1 + X_train[i][0]
        sum2_feature2 = sum2_feature2 + X_train[i][1]
        sum2_feature3 = sum2_feature3 + X_train[i][2]
        sum2_feature4 = sum2_feature4 + X_train[i][3]
        count2 = count2 + 1
mean0_feature1 = sum0_feature1/count0
mean0_feature2 = sum0_feature2/count0
mean0_feature3 = sum0_feature3/count0
mean0_feature4 = sum0_feature3/count0


mean1_feature1 = sum1_feature1/count1
mean1_feature2 = sum1_feature2/count1
mean1_feature3 = sum1_feature3/count1
mean1_feature4 = sum1_feature4/count1

mean2_feature1 = sum2_feature1/count2
mean2_feature2 = sum2_feature1/count2
mean2_feature3 = sum2_feature1/count2
mean2_feature4 = sum2_feature1/count2

#Calculating standard deviation for each feature of each class
std0_feature1 = 0
std0_feature2 = 0
std0_feature3 = 0
std0_feature4 = 0


std1_feature1 = 0
std1_feature2 = 0
std1_feature3 = 0
std1_feature4 = 0

std2_feature1 = 0
std2_feature2 = 0
std2_feature3 = 0
std2_feature4 = 0

for i in range(0,y_train.size):
    if(y_train[i] == 0):
        std0_feature1 = std0_feature1 + (X_train[i][0]-mean0_feature1)**2
        std0_feature2 = std0_feature2 + (X_train[i][1]-mean0_feature2)**2
        std0_feature3 = std0_feature3 + (X_train[i][2]-mean0_feature3)**2
        std0_feature4 = std0_feature4 + (X_train[i][3]-mean0_feature4)**2
    elif(y_train[i] == 1):
        std1_feature1 = std1_feature1 + (X_train[i][0]-mean1_feature1)**2
        std1_feature2 = std1_feature2 + (X_train[i][1]-mean1_feature2)**2
        std1_feature3 = std1_feature3 + (X_train[i][2]-mean1_feature3)**2
        std1_feature4 = std1_feature4 + (X_train[i][3]-mean1_feature4)**2
        count1 = count1 + 1
    elif(y_train[i] == 2):
        std2_feature1 = std1_feature1 + (X_train[i][0]-mean2_feature1)**2
        std2_feature2 = std1_feature2 + (X_train[i][0]-mean2_feature2)**2
        std2_feature3 = std1_feature3 + (X_train[i][0]-mean2_feature3)**2
        std2_feature4 = std1_feature4 + (X_train[i][0]-mean2_feature4)**2

std0_feature1 = std0_feature1/count0
std0_feature2 = std0_feature2/count0
std0_feature3 = std0_feature3/count0
std0_feature4 = std0_feature4/count0

std1_feature1 = std1_feature1/count1
std1_feature2 = std1_feature2/count1
std1_feature3 = std1_feature3/count1
std1_feature4 = std1_feature4/count1

std2_feature1 = std2_feature1/count2
std2_feature1 = std2_feature2/count2
std2_feature1 = std2_feature3/count2
std2_feature1 = std2_feature4/count2

#defining gaussian pdf
def gpdf(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

acc = 0
for i in range(0,y_test.size):
    a = gpdf(X_test[i][0],mean0_feature1,std0_feature1)*gpdf(X_test[i][1],mean0_feature2,std0_feature2)*gpdf(X_test[i][2],mean0_feature3,std0_feature3)*gpdf(X_test[i][3],mean0_feature4,std0_feature4)
    b = gpdf(X_test[i][0],mean1_feature1,std1_feature1)*gpdf(X_test[i][1],mean1_feature2,std1_feature2)*gpdf(X_test[i][2],mean1_feature3,std1_feature3)*gpdf(X_test[i][3],mean1_feature4,std1_feature4)
    c = gpdf(X_test[i][0],mean2_feature1,std2_feature1)*gpdf(X_test[i][1],mean2_feature2,std2_feature2)*gpdf(X_test[i][2],mean2_feature3,std2_feature3)*gpdf(X_test[i][3],mean2_feature4,std2_feature4)
    if(a>b):
        if(a>c):
            y=0
        else:
            y=2
    else:
        if(b>c):
            y=1
        else:
            y=2
    if(y == y_test[i]):
        acc = acc + 1
    print("a:",a,"b:",b,"c:",c)
accuraccy = (acc/y_test.size)*100

print("Accuracy:",accuraccy)
