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







