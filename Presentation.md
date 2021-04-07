Presentation

## Supervised Binary Classifier For IoT Data Stream 

Our project is about processing a Data Stream and fitting a Binary Classifier to predict the class label. 
We apply techniques that are different from the methods used to analyse regular stationary tabular data. 

We have met several challenges:
1. How to deal with imbalanced data set when we don’t know in advance how many samples we have.
2. How to build the model incrementally as the data comes in.
3. How to assess the performance of the models in non-stationary environment.


### Outline of the presentation

1. Introduction – Goal of the project
2. Dataset Description 
3. Data Analysis
4. Results and Discussion
5. Conclusions

###  1. Introduction – Goal of the project

Context – IoT sensors collect various telemetry data used in smart autonomous buildings.
Goal – Process the stream and build a binary classifier able to accurately predict the target label.  The target label is “light”. Our goal is to fit a model  that can predict if the light is on or off.
Applications – This prediction could be interesting in case of emergency if the sensor breaks, and we don’t receive data anymore, the light sensor might not be reliable,  for anomaly detection, for remote control.

### 2. Dataset Description
 
The data is produced by sensors arrays. 
Each sensor array emits different measurements - temperature, humidity, carbon monoxide level, motion detection, smoke detection. 

![image](https://user-images.githubusercontent.com/39202594/113806869-2121ff80-9731-11eb-87c1-c897e7a87bc6.png)

And all sensors are feeding into a central controller on one input port. 

 - Volume - 405,184 data points
 - Velocity - Data spans a period of 8 days. The average rate is 1 entry every 1.33 seconds
 - Variety - Data gathered from three arrays of IoT sensors located in  different conditions
 - Veracity – Data is accurate, no missing data, only few duplicates
 - Value – Data is useful for monitoring and control of indoor environment
 - Target class – “light” !
 

## Data Analysis

Our dataset is infinite and non-stationary. 

-	Infinite : data is produced constantly by the sensors
-	Non stationary: the distribution can change over time
-	Number of classes: 2
-	Majority class label: Light OFF

![image](https://user-images.githubusercontent.com/39202594/113768643-1134fc00-96ee-11eb-94ff-3540804938f1.png)

We have combined the results from all devices in order to obtain generalized model.

We have investigated the distribution of the class label overtime. We see that the class label is not evenly distrbuted.

![image](https://user-images.githubusercontent.com/39202594/113768759-31fd5180-96ee-11eb-9011-aa0764e0bb99.png)

The distibution of the majority vs the minority class:
![image](https://user-images.githubusercontent.com/39202594/113768970-74269300-96ee-11eb-9bfd-43a692c68574.png)

## Solution for imbalanced data set

#### 1. Class weights

We used the method **compute_sample_weight** method to adjust the class weights inversely proportional to class frequencies in the input data. With the class weights we give more emphasis of the minority class.

#### 2. Reservoir Sampling

With Reservoir Sampling we fill a preallocated buffer, called a reservoir, with uniformly sampled elements from the datastream. However, we use only data points from the majority class to fill the reservoir. We use every data point from the minority class to train the model. Once we reach a given number of training samples from the minority class, we use the reservoir to train the model with the equal number of data points from the majority class. Our goal is udersample the majority class.

[Illustration of Undersampling]
![image](https://user-images.githubusercontent.com/39202594/113810720-d2786380-9738-11eb-98fb-1b48f5cf0a75.png)



## Machine Learning Models for Classification

1. Hoeffding Tree
2. Naive Bayes

### 1. Hoeffding Tree

With conventional Decision Tree method, the training is too slow. Moreover, as the data grows indefinitely it wont fit in memory. With the CART method we need the entire data set in order to decide how to split at each node. We calculate the imputity measure by using all data points. With Hoeffding tree we dont need the entire data set, but only at a sufficiently large random subset.


## Performance Analysis

#### 1. Using holdout method.

With the holdout method we need some data to set asside for testing purposes. We use the method **train_test_split** from the scikit package to split our data set before we start the training. We make prediction on the selected test set, at regular intervals, for example every 100 data points. We can improve this method by selection a holdout test set from the stream of size **k** every **n** instances. Our results show on average high Accuracy and F-score, but results are not reliable, due to the uneven distribution of the class label:  

[Accuracy]

![image](https://user-images.githubusercontent.com/39202594/113769091-94565200-96ee-11eb-92cd-f45afbf25a55.png)

[F1-SCore]

![image](https://user-images.githubusercontent.com/39202594/113769169-ab953f80-96ee-11eb-8d50-3eca0dbcbca1.png)

#### 2. Using Predictive sequential (prequential metric)

We use this method when there is no data available for testing. In streaming environment the data comes in and we cannot split the data to a test set and a training set.
So, we used the method **"first test then train"**, i.e for each example in the stream, we make a prediction using our current model, and then we use the same data point to update the model. We count the number of correctly predicted label vs the total number of training examples which gives us the following Accuracy and F-score:

[Accuracy and F-score]

![image](https://user-images.githubusercontent.com/39202594/113769295-d384a300-96ee-11eb-86b1-77cbb2a61b38.png)

## Conclusions

1. Classification algorithms in data streams would perform well when the number of samples of each class is the same. However in reality this is rarely the case, which is a challenge. Our models are biased towards the majority class.

2. In case of data stream we need techniques that are different from the methods used to analyze regular stationary tabular data. Stream data can continuously evolve over time. Reservoir Sampling was used to sample data from the stream for training.

3. With Hoeffding Tree we do not need the entire data set to build the decision tree. We can build it incrementally as data points arrive.

4. Evaluation of  the obtained models was a challenge. We applied the following methods: Holdout of an independent test set and Prequential Error.


