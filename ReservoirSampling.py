#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skmultiflow.trees import HoeffdingTree, HoeffdingTreeClassifier
import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import pandas as pd


# In[10]:


# undersample the majority class only
# use every data point of the minority class

# 1. Accumulate N data points from the majority class in the reservoir
# 2. countMinority =0 -> start training with the minority class
# 3. When the ith>=N+1 data point arrives 
# verify if it is from the minority class
# 4. If Yes, and i< 2*N => use it for training
# 5. If No, use reservoir sampling
# 6. Did we train the model with N points from the minority class
# 7. If Yes then use the reservoir 
# to train the model with N samples from the majority class as well
# 8. empty the reservoir -> go to step1

# we apply test then train method 
# i.e we dont split to train and test set
# we use every data sample to test the model and then to train it
# calculate the prequential error

# data preparation
df=pd.read_csv("iot_telemetry_data.csv")
df["light"]=df["light"].astype(int)
df["motion"]=df["motion"].astype(int)
select_cols = ['co', 'humidity', 'lpg', 'motion','smoke','temp', 'light']
data = df[select_cols].to_numpy()

#initialization of variables
N=5 # the number of data points in the reservoir
reservoir =[]
numberCorrectPredictions=0
numberTrainingSamplesTotal =0
numberTP=0    #true positives
numberFP=0    #false positives
numberFN=0    #false negatives
result_scores=[]
result_f_scores=[]
majorityClass =0
minorityClass =1
countMajority=0
countMinority=0

#initialization of the model
ht = HoeffdingTreeClassifier()

for i in range(len(data)):
    class_label = data[i][-1]  # take the label 

    if (class_label==majorityClass):
        countMajority+=1
        if (len(reservoir) <= N-1):
            # accumulate N elements in the reservoir
            reservoir.append(data[i])
        else:
            probability=N/countMajority
            if random.random()<probability:
                m=random.randint(0, N-1)
                reservoir[m]=data[i]          
    else:
        if (len(reservoir) == N) : # we need to have N data points in the reservoir before proceeding
            # test and train the Hoeffding tree with the current data point
            countMinority+=1
            
            # take the current data point
            X=data[i][:-1]
            # X is a numpy array with shape (n_samples, n_features)
            X=np.array([X])
            
            #use the same sample to test first then to train
            y_predict = ht.predict(X)

            #train the Hoeffding Tree with the sample     
            ht=ht.partial_fit(X = X,y=np.array([class_label]))
            numberTrainingSamplesTotal +=1
#             print("training with minority***************")
#             print(i)
                 
            # calculate the current accuracy
            if (y_predict[0] == class_label):
                numberCorrectPredictions+=1
            if (y_predict[0]==1 and class_label==1):
                numberTP+=1
            if (y_predict[0]==1 and class_label==0):
                numberFP+=1
            if (y_predict[0]==0 and class_label==1):
                numberFN+=1
            current_score = numberCorrectPredictions/ numberTrainingSamplesTotal
            result_scores.append(current_score)
#             print("correct so far / total training****************")
#             print(numberCorrectPredictions)
#             print(numberTrainingSamplesTotal)
#             print(current_score)
            
            if (((numberTP + numberFP)==0) or ((numberTP + numberFN)==0)):
                current_f_score=0
            else:
                current_f_score = numberTP/(numberTP + 0.5*(numberFP+numberFN))
#             print("current_f_score************************")
#             print(numberTrainingSamplesTotal)
#             print(current_f_score)
#             print(numberTP)
#             print(numberFP)
#             print(numberFN)
            
            result_f_scores.append(current_f_score)
        
        # did we train with N points from the minority class?
        # if Yes we need to train with N points from the reservoir
        # and we will empty the reservoir 
        if (countMinority ==N):
            
            # train the Hoeffding tree with the samples from the reservoir (N from majority class)      
            for r in reservoir:
                X=r[:-1]             # take the feature columns
                class_label = r[-1]  # take the label column
                
                # X is a numpy array with shape (n_samples, n_features)
                X=np.array([X])
                
                #use the same sample to test first then to train
                y_predict = ht.predict(X)

                #train the Hoeffding Tree with the sample     
                ht=ht.partial_fit(X = X,y=np.array([class_label]))
                numberTrainingSamplesTotal +=1
#                 print("majority----------------------")
#                 print(numberTrainingSamplesTotal)

                # calculate the current accuracy
                if (y_predict[0] == class_label):
                    numberCorrectPredictions+=1
                if (y_predict[0]==1 and class_label==1):
                    numberTP+=1
                if (y_predict[0]==1 and class_label==0):
                    numberFP+=1
                if (y_predict[0]==0 and class_label==1):
                    numberFN+=1
                current_score = numberCorrectPredictions/ numberTrainingSamplesTotal
                result_scores.append(current_score)
#                 print("correct so far / total training-----------------")
#                 print(numberCorrectPredictions)
#                 print(numberTrainingSamplesTotal)
#                 print(current_score)
            
                if (((numberTP + numberFP)==0) or ((numberTP + numberFN)==0)):
                    current_f_score=0
                else:
                    current_f_score = numberTP/(numberTP + 0.5*(numberFP+numberFN))
#                 print("current_f_score__________")
#                 print(numberTrainingSamplesTotal)
#                 print(current_f_score)
#                 print(numberTP)
#                 print(numberFP)
#                 print(numberFN)
                    
                result_f_scores.append(current_f_score)
                
            # clear the reservoir after training
            reservoir=[]
            countMajority=0
            countMinority =0 
# Result
# correct so far / total training****************
# 0
# 1
# 0.0
# correct so far / total training****************
# 0
# 2
# 0.0
# correct so far / total training****************
# 1
# 3
# 0.3333333333333333
# correct so far / total training****************
# 2
# 4
# 0.5
# correct so far / total training****************
# 3
# 5
# 0.6
# correct so far / total training-----------------
# 4
# 6
# 0.6666666666666666
# correct so far / total training-----------------
# 5
# 7
# 0.7142857142857143
# correct so far / total training-----------------
# 6
# 8
# 0.75
# correct so far / total training-----------------
# 7
# 9
# 0.7777777777777778
# correct so far / total training-----------------
# 8
# 10
# 0.8
# correct so far / total training****************
# 8
# 11
# 0.7272727272727273
# correct so far / total training****************
# 9
# 12
# 0.75
# correct so far / total training****************
# 10
# 13
# 0.7692307692307693
# correct so far / total training****************
# 11
# 14
# 0.7857142857142857


# In[12]:


plt.figure(figsize=(10,5))
plt.plot(result_f_scores, label="F-Score")
plt.plot(result_scores, label="Accuracy")
plt.legend()
plt.xlabel("Number Training Samples")


# In[ ]:





# In[14]:


# take only the the firt 100 results
plt.figure(figsize=(10,5))
plt.plot(result_f_scores[:100], label="F-Score")
plt.plot(result_scores[:100], label="Accuracy")
plt.legend()
plt.xlabel("Number Training Samples")


# In[ ]:




