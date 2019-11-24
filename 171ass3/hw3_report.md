# Homework 3 report		

Ruoyan Yin

916666619

## 1. Regulized regression:

### 1.1 Comparisons between regulizing method:

The common realizing methods include ridge regression, lasso and eclastic net. All of these techniques control the amount of the coeffiients by using some penalty terms, usually the norms of the coeffient vector. Before deploy anyone of them, it is necessary to check what is difference between methods and how they perform. 

The objective function of ridge regrssion is: 
$$
\min_{w} || X w - y||_2^2 + \alpha ||w||_2^2
$$

By changing $\alpha$, ridge regressor is able to control the amount the shrinkage, and larger the $\alpha$ is, greater is the amount of shrinkage.

The obejctive function of lasso regression is:
$$
\min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}
$$

Lasso is a linear model designed to estimate sparse coefficients. By using the L1 norm of $\omega$ to strengthen the control of the size and dividing the SSE by $2n_{\text{samples}}$ to weaken the imporatance of the error, lasso will tend to choose a model with fewer non-zeros.

And the objective function of elastic net is:
$$
\min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
\frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}
$$
This method is the combination of ridge and lasso. Elastic net uses both L1 and L2 norm as penalty term and divide the total error by $2n_{\text{samples}}$.  Elastic net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

### 1.2 Decide on the predictior:

Since the data is 









 