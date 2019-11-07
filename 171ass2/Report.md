# HW 2 Report

Ruoyan Yin

916666619

---

## 1. Ourlier Detection:

The main goal in this part is to use a suitable outlier detection methode to find the outliers in the yeast data.

First, I draw the scatter matrix to briefly get the impression on outliers. 

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/pics/pair.png)

From the plot above, we can see most of the points gather in one single cluster which indicates there should not be too much outliers in this data set. Then I use different method to distinguish outliers.

### 1.1 One-class SVM

One-class SVM is a unsupervised outlier detection. It is sensitive to outliers and it is not so suitable for doing outlier detection since it tend to include most point as inliers.

However, this method performs well on novelty detection when there is no outliers in training set.

After inplement the one-class svm on this data set, I got the table as below:

| outlier | inlier |
| :-----: | ------ |
|   743   | 741    |

As indicated by the table, the number of outliers is contradictory to the preliminary analysis based on the sactter matrix, even exceeding the number of inliers. So the result is consistent to the property of the method illustrated above, one-class SVM should not be considered as a strategy of outlier detection.

### 1.2 Local Outlier Factor (LOF)

LOF is also an unsupervised outlier detection algorithm.  It measures the local deviation of density of a given sample with respect to its neighbors. And as the name indicated, it is specifically designed for the outlier detection and should perform a lot better than one-class SVM. 

Actually, I got the table as below:

| OUlier | inlier |
| :----: | :----: |
|  149   |  1335  |

Using LOF makes more sense than using one-class SVM. To make fully comparisons between method and decide on which detector to use, I then implement the isolation forest.

### 1.3 Isolation Forest

The isolation forest has a recursive structure to split data,  and so it can be represented as a forest. The idea of the algorithm is that random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

After implementing the algorithm, I get the split as the table indicates.

| outlier | inlier |
| :-----: | ------ |
|   149   | 1335   |

It seems that the number of outliers is equal to it of the result using LOF, but actually the reasult is different. Usually, isolation tree is considered to be more efficient, so I used this method to remove the outliers.

## 2. ANN Model

### 2.1 Pre-processing data

Before build up and fit my ANN model, the first thing to do is to pre-process the data.

Firsit of all, I standardize  all the idependent variables by using StandardScaler.

Then, to make the data fit the structure of the model, I transformed the categories into numerical lables and then encoded it by one-hot encoding.

### 2.2 Fit the model

As we can see in the loss and weight plost, the curves of  loss for the whole model and the weights for the 'CYT' node converge after1000 batches fed which indicates the model get stable. However after some point, the weights is getting bigger which may indicates the existence of overfitting. 

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/colab_pics/loss.png)

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/colab_pics/weight.png)

By tracking the error of the class 'CYT', I got the plot as below. The error didn't change for the most of the epochs and got stuck at around 0.68 for training error and 0.66 for test error. One of possible reasons for the result is that the model was trying to predict all the input as the 'CYT' class since this class has the largest size and by predicting any input data as 'CYT' can help to reach to the minimum of the MSE. Then I tried to make prediction on the training data using the trained model, the result was excatly the same as the inference, that is I got all the prediction as 0 which stands for the 'CYT' class.

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/colab_pics/error2.png)

## 3. Train ANN on all the sample

After traing the model on all the samples, I got the error as 0.68, not too much departing from the one in part 2. And the weights for the 'CYT' node locating at  output layer is $[-2.05184, -1.379138, -1.3926986, -1.3907589]$.

So the activation function for the node is sigmoid: $a_1^{(4)} = \frac{1}{1+e^{-z}}$

Where: $z = w_1 a^{(3)}_{11} + w_2 a^{(3)}_{21} +w_3 a^{(3)}_{31} + w_0$ 

and $w_1 = -1.379138, w_2 = -1.3926986, w_3 = -1.3907589, w_0 = -2.05184.$

## 4. Back propagation



## 5. Grid Search

By changing the number of layers and the number of nodes of each hidden layer iteratively, I finally get the error matrix as required as below:

| layer\node |     3      |     6      |     9      |     12     |
| :--------: | :--------: | :--------: | :--------: | :--------: |
|     1      | 0.66519824 | 0.69823789 | 0.66519824 | 0.69823789 |
|     2      | 0.66519824 | 0.66519824 | 0.69823789 | 0.66519824 |
|     3      | 0.66519824 | 0.66519824 | 0.69823789 | 0.66519824 |

And I also get the plot showing how the overall error changing with epochs and a plot showing the time taken to fit the model.

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/colab_pics/grid_search1.png)

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/colab_pics/time.png)

As indicated by the error plot, most of the models reach the error aroud 0.66, and the top 3 combinations that converge are:

- Layer = 3, node = 6
- Layer = 3, node = 12
- Layer = 1, node = 9

However, the time that was used to fit the model varies a lot . Basically, more layers or nodes the model have meaning more complex the model is, and more time is supposed to take according to the second plot. Under this idea, we don't want to chose the complicated model just because of a slightly better performance. So combining the error plot and the time consumption plot, I will say the best configuration is layer = 3, node = 6 since it converges fast, and the model is not too conplex that will take a lot time to train. 

## 6. Make prediction

From part 5, I made the decision to use the model with 3 hidden layers and 9 nodes in each layer. Then build a column vector according to the data given in the document. After that I standardize it and finally put it into the model get the prediction as 0.

## 7.Change the activation function and loss

### 7.1 Grid Search

Similarly to part five, to decide the suitable model for the problem after changing the activation functions and loss, I applied grid search first to see which model to deploy. I still draw the plot showing how the error changes with epochs. 

Showed by the figure, after about 200 epochs, all the errors are fluctuating around andthe one reaches to the minimal error is the model having 3 hidden layes and 6 nodes in each layer.

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/colab_pics/grid_search2.png)

### 7.2 Re-train the model

Then I re-train the model and get the testing and traing error for the model. And finally, the error fluctuate around 0.3. 

![](/Users/yry/Dropbox/Courses/ecs171/Projects/171ass2/colab_pics/error3.png)



