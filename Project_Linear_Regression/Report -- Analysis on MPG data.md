##Homework Report -- Analysis on MPG data

Ruoyan Yin

916666619

----

#### 1. Catagorize the data

Before set the threshold for the mpg level, it is neccessary to have some basic statistics of the data. So I have this:

| statistics | value |
| :--------: | :---: |
|   count    |  392  |
|    mean    | 23.45 |
|    std     | 7.80  |
|    min     | 9.00  |
|    25%     | 17.00 |
|    50%     | 22.75 |
|    75%     | 29.00 |
|    max     | 56.60 |



From the table, we can see the quantiles listed. Ideally, if we just use corresponding value to be the threshold, then the seperation should be equally-sized.

 However, as we can expect, there are some indentical values in mpg. It will make no sense that tag one as some level and tag the other same one to other level just on order to get the equally-cutted bin. In this case, the size of each bin can vary a little. Actually, after using pandas built-in method `qcut` (referring to quantile cut), I get the size of each bin as below:

|   level   | size |
| :-------: | :--: |
|    low    |  99  |
|  medium   |  97  |
|   high    | 101  |
| very high |  95  |

----

#### 2. Plot the scatter matrix

I got a scatter matrix as below:

​                                                             Scatter Matrix

![](/Users/yry/Desktop/Courses/ecs171/171ass1/scatter_matrix.jpg)

From the graph, regarding the four mpg catagories, I will say the most informative pair is year and weight.

The reason is that even though most pairs in the middle 5*5 entries of the matrix can somehow seperate the catagories, most of them red points(very_high) overlap a lot with green points(high) . The combination of year and weight is the only pair that the 4  groups of points overlap  least.

----

#### 3. My linear regression solver

##### 3.1 `lm_solver` documentation

In this part I build up my linear regression solver as a new class `lm_solver`.  This solver can deal with multiple variables and higher order. Methods and attributes of the class are listed below:

| Attributes   | description                             |
| :----------- | :-------------------------------------- |
| intercept    | Whether the model have intercept or not |
| order        | The order of the model                  |
| n            | Sample size                             |
| p            | Number of independent variables         |
| coef_        | The coefficients of the model           |
| h            | Design matrix                           |
| fitted_value | Fitted value                            |
| residual     | Residual                                |
|              |                                         |

| Methods     | Description                             |
| ----------- | --------------------------------------- |
| standardize | Standardize the input data              |
| fit         | Fit the linear model using given data   |
| get_mse     | Get the MSE of the fitted model         |
| predict     | Make prediction using the fitted model  |
| plot_       | Plot the fitted value and observations. |

##### 3.2 Test my `lm_solver`

I generate $Y=3+ x_1 ^2 + 2x_2 + e$, where e is random error. And as result I have the coefficient as: 3.222, 0.003, 0.998, 2.01, 0. 

----

#### 4. Fit the model using the `lm_solver`

##### 4.1 Split the data

Before fit the model, I split the data into 2 sets, training set and testing set. Furthermore, to keep the distribution of the splitted data the same as the whole data set, I apply stratified shuffle spilt. So I can have two data sets has similar distribution which is close to the original distribution. And hence the structure of the data won't be influenced by sampling and the effect the regressor should be stable.

After the splitting, we got the ratio as below:

| catagory  | Whole set | Training | Testing  |
| --------- | --------- | -------- | -------- |
| Low       | 0.252551  | 0.26     | 0.253425 |
| Medium    | 0.247449  | 0.25     | 0.246575 |
| High      | 0.257653  | 0.25     | 0.256849 |
| Very_high | 0.242347  | 0.24     | 0.243151 |

So from the table, after performing stratified split, since we set the categories to have the equal size, the traing and testing set have the similar distribution. So ideally, the result should be stable or at least less biased.

##### 4.2 Fit the model

After fit the model, I these 4 plots which stands for the models with order from 0 to 3.

![accel](/Users/yry/Desktop/Courses/ecs171/171ass1/accel.png)

![cyl](/Users/yry/Desktop/Courses/ecs171/171ass1/cyl.png)

![displ](/Users/yry/Desktop/Courses/ecs171/171ass1/displ.png)

![hp](/Users/yry/Desktop/Courses/ecs171/171ass1/hp.png)

![weight](/Users/yry/Desktop/Courses/ecs171/171ass1/weight.png)

![yr](/Users/yry/Desktop/Courses/ecs171/171ass1/yr.png)

![origin](/Users/yry/Desktop/Courses/ecs171/171ass1/origin.png)

Just like what we can see from these 7 plots, using whther displacement, weight, horsepower or acceleration can get a resonable curve. So to be more accrutly measure which model performs best, I list 2 tables containing training and testing MSE.

Training MSE:

| Variable | order=0 | order=1 | order=2 | order=3    |
| :------: | :-----: | :-----: | ------- | ---------- |
|   cyl    | 62.323  | 26.528  | 26.524  | 25.058     |
|  displ   | 62.323  | 25.306  | 22.880  | 22.685     |
|    hp    | 62.323  | 26.255  | 22.560  | 22.557     |
|  weight  | 62.323  | 21.644  | 19.509  | 19.506     |
|  accel   | 62.323  | 52.825  | 51.506  | 51.271     |
|    yr    | 62.323  | 38.818  | 34.206  | 33.888     |
|  origin  | 62.323  | 42.002  | 41.244  | 190349.954 |

Testing MSE:

| Variable | order=0 | order=1 | order=2 | order=3    |
| :------: | :-----: | :-----: | ------- | ---------- |
|   cyl    | 60.271  | 23.299  | 23.242  | 21.168     |
|  displ   | 60.271  | 20.188  | 17.652  | 17.741     |
|    hp    | 60.271  | 23.299  | 18.100  | 18.074     |
|  weight  | 60.271  | 17.668  | 16.648  | 16.644     |
|  accel   | 60.271  | 49.376  | 48.949  | 49.105     |
|    yr    | 60.271  | 41.080  | 40.696  | 41.676     |
|  origin  | 60.271  | 41.191  | 40.411  | 205307.759 |

----

From the table above, when using weight as variable to fit a third order polynomial can reach the minimal MSE. Also, except the 0th order, weight always has least test MSE, so it will be reasonable to say weight should be the most informative variable.

#### 5. Second order polynomial of 7 variables

After modifying my linear solver and regressing on the all 7 variables from order 0 to order 2, I list the all the results in the following table:

| Order | No. of coef | training Mse | testing mse |
| :---: | :---------: | :----------: | :---------: |
|   0   |      1      |   60.9075    |   60.3452   |
|   1   |      8      |   11.1387    |   10.3263   |
|   2   |     15      |    7.6489    |   6.4954    |

Not suprisingly, the second order model has the best performance with respect to tetsing MSE.

----

#### 6. Logistic Regression:

In this part, I build up the logistic regressor using sci-kit learn package. Still, before regressing, I have to spilt the data in a manner as before that is stratified split. 

After that, I tried fit the model using default max_iteration setting which is 100, however, the sample size retricted the converging speed. So have to raise it up to 5000. And get my prediction.

Then, using sklearn built-in method, I get the score of the model on testing set as **0.8**.

Below is how sklearn calculate the score of logistic regressor according to its documentation.

> Returns the mean accuracy on the given test data and labels.
>
> In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.

According to the documentation, the score of the model means **80%** of the prediction hit the target.

----

#### 7. Logistic Regression combined with normalization

After apply min-max normalization, I fitted the model and get the score again. This time the score is **0.81**, slightly higher than before.  But the other aspect of performance which is coverging speed has increased a lot. This time the model converged using default max_iter.

----

#### 8. Making prediction

This question is pretty straightforward. According to the description, I the data point to be predicted is:

$$a = (4, 400, 150, 3500, 8, 81, 1)$$

Then I normalized the point and got:

$$a' = (0, 0.856, 0.581, 0.564, -0.037, 0.917)$$

After that put the point into 2 models and get the predictions:

|        model        | prediction |
| :-----------------: | :--------: |
|  Liner regression   |   22.559   |
| Logistic regression |   Medium   |

So the two result is actually consistent based on my method of factorizing the mpg data.
