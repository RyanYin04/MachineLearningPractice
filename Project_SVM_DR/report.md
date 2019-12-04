# Homework 3 Report		

Ruoyan Yin

916666619

## 1. Regulized regression

### 1.1 Comparisons between regulizing method:

The common realizing methods include ridge regression, lasso and eclastic net. All of these techniques control the amount of the coeffiients by using some penalty terms, usually the norms of the coeffient vector. Before deploy anyone of them, it is necessary to check what is difference between methods and how they perform. 

#### 1.1.1 Ridge

The objective function of ridge regrssion is: 
$$
\min_{w} || X w - y||_2^2 + \alpha ||w||_2^2
$$

By changing $\alpha$, ridge regressor is able to control the amount the shrinkage, and larger the $\alpha$ is, greater is the amount of shrinkage.

#### 1.1.2 Lasso

The obejctive function of lasso regression is:
$$
\min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}
$$

Lasso is a linear model designed to estimate sparse coefficients. By using the L1 norm of $\omega$ to strengthen the control of the size and dividing the SSE by $2n_{\text{samples}}$ to weaken the imporatance of the error, lasso will tend to choose a model with fewer non-zeros.

#### 1.1.3 Elastic Net

And the objective function of elastic net is:
$$
\min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
\frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}
$$
This method is the combination of ridge and lasso. Elastic net uses both L1 and L2 norm as penalty term and divide the total error by $2n_{\text{samples}}$.  Elastic net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

### 1.2 Decide on the predictior:

As the data itself indicates, there are more than 4,000 attributes. So it won't be ideal to use ridge. And to keep advancements from both lasso and ridge, I decided to us elastic net and tune the parameters by using grid search.

The range of $\alpha (\text{the penalty term})$ I searched on is $(0.5, 1, 2, 4, 8, 16)$, and the range of $\rho(\text{L1 norm ratio})$ I searched on is $(.1, .25, .5, .75, .9, 1)$. 

To make full use of data, 5-fold cross validation is implemented here. Then I got a table listing all the 5-fold cross validation generalized error as below:

| $\rho$\ $\alpha$ |   0.5    |    1     |    2     |    4     |          8          |    16    |
| :--------------: | :------: | :------: | :------: | :------: | :-----------------: | :------: |
|       0.10       | 0.055702 | 0.055702 | 0.055771 | 0.055743 | <u>**0.053729**</u> | 0.056487 |
|       0.25       | 0.055702 | 0.055702 | 0.055702 | 0.055779 |      0.055749       | 0.054794 |
|       0.50       | 0.055702 | 0.055702 | 0.055702 | 0.055702 |      0.055749       | 0.054794 |
|       0.75       | 0.055702 | 0.055702 | 0.055702 | 0.055702 |      0.055702       | 0.055768 |
|       0.90       | 0.055702 | 0.055702 | 0.055702 | 0.055702 |      0.055702       | 0.055782 |
|       1.00       | 0.055702 | 0.055702 | 0.055702 | 0.055702 |      0.055702       | 0.055779 |

From the table, the best combination is $\alpha$ to be 8, and $\rho$ to be 0.1. This model will only include 8 attributes and the column indices of them in the datafram are: $\text{152, 159, 400, 723, 2409, 2576, 2718, 3088}$. The corresponding gene expression are: ‘, 'b4139', 'b1266', 'b0853', 'b3197', 'b1587', 'b3900', 'b2731'.

## 2. Methodology of Bootstrap

The idea of bootstrap is that when the data size is small and we are not able to abtain sufficient samples, we can sample from data repeatedly with replacement. 

To get the confidence interval using bootstrap, there are some different methods:

The first one is using bootstrap to calculate sample stastistics to estimate the population statistics. Then based on the nomality asuumption of error, we can get the t-statistic of the prediction and finally calculate the confidence interval.

The second one is fit the model by the sub-sample selected from the whole samples with replacement which is actually using bootstrap method and then make prediction using the fitted model. After the bootstrap process finished, we have a sample distribution on the prediction. Then sort all the predictions and the confidence interval can be estimated as:
$$
[pred_{\alpha/2},\quad pred_{1-\alpha/2}]
$$
where $\alpha$ is the confidence level.

I will chose the second method, the quantile method, as my way to get the bootstrap confidence interval.

## 3. Make prediction using bootstrap method

By setting the number of sampling to be 500 and the sample size of each batch to be 100, I got the 95% confidence interval as:
$$
(0.3541632925954786,\quad 0.4331288649549101)
$$

## 4. SVM model

The goal here is to build 4 models to make prediction on strain type, medium type, environmental stress and gene perturbation. ROC and PR curves are also required to visualize the performance of the model. As a supplemental evidence, AUC and PRAUC are desired as well.

To compress the size of the model, I did not use all the attributes in the model as predictors, I just use the 8 attributes selected in the elastic net model instead.

Before get all the plots, there are two problems should be mentioned. The first one is that usually ROC curve and PR curve are designed to asses the binary classifiers. When dealing with more than 2 classes, additional transformation of data is required. The way I transform and calculate the curves is re-encode the classes as integers and using micro average. 

Finally, I get all the plots as below:

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Strain_part4.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Medium_part4.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Evrn_part4.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Gene_part4.png) 

Here all the SVM models used 'rbf' kernel.  From the ROC curves, the models performs decently. The average AUCs for strain type, medium type, environmental perturbation, gene perturbation are 0.91, 0.92, 0.86, 0.93 respectively.

However, by checking the data, I found out that it was not balanced. So using PR curves and PRAUC should be more appropriate. And this time, such a simple SVM model shows some deficiencies. The average of PRAUCs for each model are 0.72, 0.61, 0.59, 0.79. 

## 5. Composite SVM

### 5.1 Composite model

To get a composite SVM model, I firstly concatenate tow columns into 1 column and use the new column as a new response. Then train the SVM model on the dataset again. I got the AUC and PRAUC as 0.97 and 0.78 this time. And the ROC, PR curves are showed as below.

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_combined_part5.png)

### 5.2 Evaluate composite model against seperate models

To get the overall ROC and PR curve as well as AUC and PRAUC correspondingly for two seperate classifier, the process has been changed.

Firstly, I fit the model seperately and calculate the score on the test set, i.e. probability. And then to get the combined probability, I use the equation:
$$
P(AB) = P(A)P(B)
$$
Finally, using macro average to calculate the men AUC and PRAUC. And get the plot as below:

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/part5_seperate_model.png)

The mean AUC is 0.98, PRAUC is 0.68. So the AUC is similar to it of composite model, but the PRAUC is much less. And since the data is not balanced, so PRAUC  will be more persuasive. So it might me more reasonable to use composite model instead two seperate ones.

### 5.3 Evaluate composite model against baseline model

In order to tell if the model is really working, it is neccessary to check how the baseline model perfoms. In this case, I create a dummy model using 'most frequent' strategy that is the model will always make a prediction the same as the element that appears most in the data. And then I have the AUC and PRAUC for the dummy model as: 0.57, 0.20 which are much smaller than the ones from composite model.

So it can be concluded that the SVM model did work better than random guess.

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_combined_part5_dummy.png)

## 6. Dimensionality reduction

Before the way I use to reduce the dimensionality is appying elastic net. In this part I will use differen mothods to realize this target.

### 6.1 PCA

The first method is **Principal Component Analysis(PCA)**. After compress and transform all the 4,495 attributes by using PCA, I draw the scatter plot of the transformed data. The color of the point indicates the class that the point belong to.

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/perfomance_for_PCA.png) 

From the plots, after implement PCA on the data, the data still spread out. So it can be expected that the performance of the SVM models combined with PCA will not be improved with respect to the AUC or PRAUC.

### 6.2 T-SNE

T-SNE is another method to dimensionality in a statistical way instead of mathematical one. Similarly, I also draw the scatter plot after transformed the data.

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/perfomance_for_TSNE.png)

The scatter plot acquaired by applying T-SNE shows a better performance that the dominant label(label that appear most in data) is clustered togther. And the minor labels are spreaded. So the application of T-SNE should be helpful to imporve the SVM models.

## 7. Retrain SVMs in part 4 using PCA and T-SNE

Recall in part 4:

- Average AUCs: 0.91, 0.92, 0.86, 0.93
- Average PRAUCs: 0.72, 0.61, 0.59, 0.79

### 7.1 Use PCA

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Strain_part7_pca.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Medium_part7_pca.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Evrn_part7_pca.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Gene_part7_pca.png)



Now after PCA:

- Average AUCs: 0.94, 0.94, 0.88, 0.97
- Average PRAUCs: 0.74, 0.62, 0.65, 0.87

There are some slight improvements on the scores. This may because PCA use all the attributes to form a new set of predictors so that it will contain more information than just selecting a subset of all the attributes separately.

### 7.2 Use T-SNE 

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Strain_part7_tsne.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Medium_part7_tsne.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Evrn_part7_tsne.png)

![](/Users/yry/Dropbox/Courses/ECS171/Projects/Project_SVM_DR/pics/ROC_PR_Gene_part7_tsne.png)

Now after T-SNE:

- Average AUCs: 1.00(rounded up), 0.99, 0.91, 0.97
- Average PRAUCs: 0.99, 0.95, 0.79, 0.92

As we can see there is a significant improvement. And the SVM model for Strain type are almost perfect.

So the conclusion is that the best pre-processing aprroch for this data set is T-SNE method.

