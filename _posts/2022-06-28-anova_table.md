---
layout: post
title: MLE, my notes on estimation
published: false
comments: true
---

In my stat's class, we are using Larsen's 6th edition of `An Introduction to Mathematical Statistics and Its Applications.` This week, we are doing chapter 5, estimators. If you want a good rundown of MLE, there are tons of other resources at the end of this post.

# Likelihood Function v. Probability Function
Last semester, we did a lot of exploration with the [iris dataset](https://www.kaggle.com/datasets/uciml/iris). Briefly, the iris dataset contains data on 150 iris flowers. There are 3 different species of flowers (50 of each iris species) and each flower has 4 measurements/features (petal width/length, sepal width/length). We played with different ways to cluster the iris by species by looking at the more distinguishable features. *Spoiler, petal length and petal width varies the most between iris species!*

One clustering method we used was Gaussian Mixture Models (GMM). The whole dataset was the sum of 3 different iris species so our GMM was the sum of 3 normal distributions. To write the normal distribution for each of the iris species, all you need is the mean and standard deviation for each species. Lastly, you use the GMM to calculate the likelihood of the each data of being each of the 3 species and sort the data into the species with the highest likelihood.

One could simply calculate parameter values for the species' distributions but we took it one step more and wanted to find the *best* estimators for mean and standard deviation! We used expectation maximization, a method that uses the data to calculate the conditional probability of the species cluster using estimated parameters. The algorithm cycles, modifying the parameters until the probability converges.

$$P(species|data)=\frac{P(species) * \left(pdf \text{ of species with estimated parameters}\right)}{P(data)}$$

![Iris dataset: clustering with GMM and EM](../img/GMM_EM.png)

While I was writing up the homework, I found myself interchanging "probability" and "likelihood." I had taken stats before, but even so, I could not quite remember the distinction between likelihood and probability!

To put shortly:

> Likelihood function: how likely is the hypothesis for varying parameter values given this data?
> 
> Probability function: probability for the data, given the hypothesis

```python
if (python code):
  put it here
```
# $$\hskip-0mm\boldsymbol{\hat{\hskip1mm\theta}}$$
Additionally, we want to find the parameter value that maximizes the likelihood. In this example, we numerically calculated the likelihood for different estimators for mean and standard deviation. Each iteration, we refined the parameters until the change in previous and current likelihood was zero.

To maximize $$\theta$$ empirically, we find where the change in the likelihood function is zero, i.e., where $$ \frac{d}{d\theta} \mathcal{L}(\theta)=0$$

Likelihood is the probability of the entire dataset for a specific parameter value. So to write the likelihood function $$\mathcal{L}(\theta)$$, we multiply the probability of each data for variable $$\theta$$.

$$\mathcal{L} (\theta) = \prod_{data}^{\infty} pdf$$ (data; $$\theta$$)

Once you have a likelihood function, you can use some logarithm magic to separate the product into sums to simplify the derivative process.

# When $$\mathcal{L} (\theta)$$ does not have a finite maximum

If the likelihood function does not have a finite maximum, we cannot calculate $$\hat{\theta}$$. In this case, we can use order statistics.

### Order Statistics

I just learned about order statistics. Basicially, you sample $$n$$ times from a distribution $$Y$$ and then order the sample $$y_i$$'s in ascending order:

$$ y_1, y_2, y_3, ...y_n \quad \text{ for }y_i \le y_{i+1} $$

We can build a new distribution for the probabilities of the $$i$$th largest number in the sample, $$Y ^{\prime} _{i}$$. For a sample of size $$n$$, the distribution of the smallest value in  is $$Y^{\prime} _{min}$$, the second smallest is $$Y ^{\prime} _{2} $$,...the distribution of largest value in a sample $$n$$ would be $$Y ^{\prime} _{n}$$.

In this case, $$ \hat{ \theta}$$ will be either $$y_{min}$$ or $$y_{max}$$ depending on the likelihood function. In the example from Larsen below, we can see the $$\mathcal{L} (\theta)$$ is maximized by $$y_{min}$$.

![Larsen Figure 5.2.1](../img/Larsen_fig5_2_1.png)

# A simpler method: Method of Moments

The first moment is the expected value which is also the average $$\bar{y}$$. We can calculate $$\bar{y}$$ in terms of $$\theta$$. Solving for $$\theta$$, we get the maximum likelihood parameter:

$$E[Y] = \bar{y} = \int_{-\infty} ^{\infty} y * f_Y(y; \theta) dy$$




```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load dataset
'''
TABLE B.9 Pressure Drop Data

y: Dimensionless factor for the pressure drop through a bubble cap
x1: Superficial fluid velocity of the gas (cm/s)
x2: Kinematic viscosity
x3: Mesh opening (cm)
x4: Dimensionless number relating the superficial fluid velocity of 
the gas to the superficial fluid velocity of the liquid
Source: “A Correlation of Two-Phase Pressure Drops in Screen-plate Bubble Column,”
by C. H. Liu, M. Kan, and B. H. Chen, Canadian Journal of Chemical Engineering, 71, 460–463.
'''

col_list = ['y','x1','x2','x3','x4']
DF=pd.read_csv('../input/b9-montgomery/data-table-B9.XLS - Sheet1.csv', usecols=col_list)
print(DF.head())

# Plot data
from mpl_toolkits.mplot3d import Axes3D

x = DF['x1']
y = DF['x2']
z = DF['y']

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(x, y, z, color='red')

# Add axis labels
ax.set_xlabel('fluid velocity', fontweight ='bold')
ax.set_ylabel('kinematic viscosity', fontweight ='bold')
ax.set_zlabel('pressure drop', fontweight ='bold')

df=DF.sample(n=40,replace=False)

print(f'index of 40 random rows:\n')
index=df.index
for i in index:
    print(i)
    
# Convert dataframe to np array
X=df[['x1', 'x2', 'x3', 'x4']].to_numpy()
X = np.append(np.ones((len(X), 1)), X, axis=1)   # add a column of ones for the constant beta_0
y=df[['y']].to_numpy()

# Calculate estimators B_hat
## B_hat = inv(X.T * X) * X.T * y
Xt=X.T
B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xt, X)),Xt),y)

print('\nestimators for beta: ')
print(f'b0-hat: {B_hat[0]} \nb1-hat: {B_hat[1]} \nb2-hat: {B_hat[2]} \nb3-hat: {B_hat[3]} \nb4-hat: {B_hat[4]}')

# Plot model with estimated parameters + data

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

x = np.linspace(0, 10, 100)
Y = np.linspace(0, 120, 100)
xx, yy = np.meshgrid(x, Y)
z = B_hat[0] + B_hat[1]*xx + B_hat[2]*yy

# plot model estimate (plane)
ax.plot_surface(xx, yy, z, alpha=0.20)

## plot data (scatterplot)
x = X[:,1]
YY = X[:,2]
# z = y
ax.scatter3D(x, YY, y, color='red')

ax.set_xlabel('fluid drop', fontweight ='bold')
ax.set_ylabel('viscosity', fontweight ='bold')
ax.set_zlabel('velocity', fontweight ='bold')
plt.title(f'y-hat = {round(float(B_hat[0]),3)} + {round(float(B_hat[1]),3)}*x1 + {round(float(B_hat[2]),3)}x2 + {round(float(B_hat[3]),3)}*x3 + {round(float(B_hat[4]),3)}*x4 ')
plt.show()

    
class aov_table:
    def __init__(self, X, y, B_hat, nregressors, intercept):
        self.X = X
        self.y = y
        self.B_hat = B_hat
        self.p = np.shape(self.B_hat)[0]
        self.k = nregressors
        self.ybar = sum(self.y)/len(self.y)
        if intercept==False:
            self.ybar=0
        else:
            self.ybar = sum(self.y)/len(self.y)
        
    def y_hat(self):
        y_h=np.matmul(self.X,self.B_hat)
        #print('Y_hat')
        #print('   y_hat = X * B_hat')
        return y_h
    
    def resid(self):
        y_resid = self.y-np.matmul(self.X,self.B_hat)
        #print('Residuals')
        #print('   e = y - X * B_hat')
        return y_resid
    
    def SS_e(self):
        # section 3.2.4, eq 3.16
        # eT * e
        first_mult= np.matmul(self.B_hat.T,self.X.T)
        ssres = np.matmul(self.y.T,self.y) - np.matmul(first_mult,self.y)
        #print('Sum of Square of Residuals SS_res')
        #print('   SSE = SS_res = e.T * e')
        #print('        where e = y - X * B_hat')
        return float(ssres)
    
    '''def SS_r(self):
        # equation from 3.24
        # b-hat.T * X.T * y - sumsquare(y)/n
        firstpart=np.matmul(self.B_hat.T,self.X.T)
        first_half=np.matmul(firstpart,self.y)
        y_part = sum(self.y)*sum(self.y)/len(self.y)
        ssr = first_half-y_part
        return float(ssr)'''

    def SS_r(self):
        # equation from appendic C.3.1
        # (y-hat - 1*ybar).T * (y-hat - 1*ybar)
        ones=np.ones((len(self.y),1))
        #self.ybar=sum(self.y)/len(self.y)
        ssr=np.matmul((self.y_hat()-ones*self.ybar).T, (self.y_hat()-ones*self.ybar))
        return float(ssr)
    
    def SS_T(self):
        # equation from C.3.1 (pg 581)
        # (y-y_bar)T * (y-y_bar)
        self.ybar = sum(self.y)/len(self.y)
        sst = np.matmul(self.y.T,self.y) - sum(self.y)*self.ybar
        #print('Sum of Square Total SST')
        #print('   SST = (y-y_bar)T * (y-y_bar)')
        #print('       = SS_r + SS_e')
        return float(sst)
    
    def MS_res(self):
        # equation 3.17
        # SS_res / (n-p), p=#params
        n=len(self.X)
        MSRES=(self.SS_e())/(n-self.k-1)
        #print('residual Mean Square MS_res')
        #print(f'   MS_res = SS_res / (n-p) = {float(MSRES)}')
        return float(MSRES)
    
    def MS_r(self):
        # SS_r / (n)
        n=len(self.X)
        MSR=(self.SS_r())/(self.k)
        #print('residual Mean Square MS_res')
        #print(f'   MS_r = SS_r / (k) = {float(MSR)}')
        return float(MSR)
    
    def F(self):
        # pg 85
        # MS_r / MS_res
        n=len(self.X)
        MSRES=(self.SS_e())/(n-self.k-1)
        MSR=(self.SS_r())/(self.k)
        f = MSR / MSRES
        return float(f)
    
    def R2(self):
        # SSR/SST
        r2=self.SS_r()/self.SS_T()
        return float(r2)
    
    def R2_adjust(self):
        # eq 3.27
        # 1 - SS_res/(n-p-1) / SST/(n-1)
        # R2 adjusted penalizes us for adding too many parameters
        n=len(self.y)
        num=self.SS_e()/(n-self.k-1)
        denom=self.SS_T()/(n-1)
        r2a = 1 - num/denom
        return float(r2a)
    
    def summary(self):
        n=len(self.y)
        import scipy.stats
        f_crit = scipy.stats.f.ppf(q=1-.05, dfn=self.k, dfd=n-self.k-1)
        MSRES=(self.SS_e())/(n-self.k-1)
        MSR=(self.SS_r())/(self.k)
        f = MSR / MSRES
        print(f'''
        Source of variation   Sum Sqaures    df   Mean Square
        ---
        Regression            SSR              k       MS_R
        (y-hat-y-bar)^2       {round(self.SS_r(),2)}          {self.k}       {round(self.MS_r(),2)}
        
        Residual              SSres            n-k-1  MS_res
        errors^2              {round(self.SS_e(),2)}             {n-self.k-1}     {round(self.MS_res(),2)}
        
        TOTAL                 SST              n-1
        (y-y_bar)^2           {round(self.SS_T(),2)}          {n-1}
        
        ---
        
        F-statistic           F0              F_crit(alpha=0.05, df={self.k}, {n-self.k-1})
        MS_R / MS_resid       {round(f,2)}            {round(f_crit,2)}
        ''')

### PART A
print(f'''3a
Perform a thourough linear regression analysis of these data to cover
- significance of regression: F-test for overall model adequacy
- regression sum of squares: df = k
- residual sum of squares: df = n-k-1
''')

print(f'Significance of regression, sum of squares, degrees of freedom')
dataset=aov_table(X,y,B_hat, 4,True)
dataset.summary()


print(f'''
3b
Significance of each regressor using the t-test,
95% CI for regression coefficient, and
estimated correlation between regression coefficients''')


print(f'''I wrote the function meanCI() which
- calculates t0 and t_crit for a regressor and rejects with a significance of alpha
- constructs a (1-alpha)% CI around the coefficient estimator
Here, we evaluate the null hypothesis at a 0.05 significance level
and create 95% CI for the estimate.
''')

# Calculate 95% CI for beta_1

'''
beta1-hat +- (t*) (sqrt(sigma-hat^2 Cjj) ) 
sigma-hat^2 = MS_res = dataset.MS_res()
Cjj = inv(X.T * X)
'''

def meanCI(X,beta,j,alpha):
    
    import scipy.stats
    
    Cjj=np.linalg.inv(np.matmul(X.T,X))[j][j]
    sigma_hat_2 = dataset.MS_res()
    se_Beta_j = np.sqrt(sigma_hat_2*Cjj)
    beta_hat = beta[j]
    degreefreedom=len(X)-4-1
    t_crit = np.abs(scipy.stats.t.ppf(q=1-alpha/2,df=degreefreedom))
    
    t0=float(beta_hat)/se_Beta_j
    rejecto = np.abs(t0)>t_crit
    conclusion = 'reject H0'
    if rejecto == False:
        conclusion = 'fail to reject H0'
    
    lower = float(beta_hat - t_crit * se_Beta_j)
    upper = float(beta_hat + t_crit* se_Beta_j)
    
    print(f'''
    --------------------------------------------
    Hypothesis test
    --------------------------------------------
    H0: beta{j} = 0
    H1: beta{j} != 0

    beta_hat{i} = {round(float(beta_hat),3)}
    se(Beta{i}) = {round(se_Beta_j,3)}
    
    t0 = beta_hat/se(beta_hat) = {round(t0,3)}
    t_crit = {round(t_crit,3)} for alpha = {alpha}, df = {degreefreedom}
    
    |t0| > t_crit is {rejecto} ---> {conclusion}''')
    
    print(f'''
    --------------------------------------------
    Confidence Interval
    --------------------------------------------
    {1-alpha} Confidence interval for Beta{j}
    beta_hat +/- t_crit * se(Beta_j)
    
    lower: {round(lower,3)}, upper: {round(upper,3)}
    ''')

    return conclusion
    
    
for i in range(5):
    print(f'................................................\n')
    print(f'REGRESSOR x{i}\n')
    print(f'{meanCI(X,B_hat,i,0.05)}\n')
 
# USING SKLEARN TO CHECK values
'''
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# SkLearn
## Using sklearn's linear_model.LinearRegression method which uses "plain Ordinary Least Squares (scipy.linalg.lstsq)"
# https://codeburst.io/multiple-linear-regression-sklearn-and-statsmodels-798750747755
from sklearn import linear_model
print('sklearn: linear_model.LinearRegression')

reg = linear_model.LinearRegression()
reg.fit(df[['x1', 'x2','x3','x4']].to_numpy(),y)
print(f'b0-hat: {reg.intercept_}')
print(f'bj-hat: {reg.coef_}\n')


X_test = df[['x1', 'x2','x3','x4']].to_numpy()
y_test = y

X_test_constants = sm.add_constant(X_test)
est = sm.OLS(y_test, X_test_constants)
est2 = est.fit()
print(est2.summary())
'''

print(f'''
3b cont
Correlation between each regression coefficient.
I calculate the correlation matrix W'W'
where the wij element is calculated via eq 3.59.
''')

n=len(df)
df.reset_index(inplace = True, drop = True)
W = pd.DataFrame(columns = ['x1','x2','x3','x4'])

for j in ['x1','x2','x3','x4']:
    w=[]
    denom=0
    xj_mean=np.mean(df[j])
    for i in range(n):
        num=df[j][i]-np.mean(df[j])
        denom= denom + (df[j][i]-xj_mean)*(df[j][i]-xj_mean)
        w.append(num)
    w=w/np.sqrt(denom)
    W[j]=w

correlation_coeff_matrix = np.matmul(np.array(W).T,np.array(W))
print(correlation_coeff_matrix)

print(f'''
3d
Linear Regression analysis excluding the intercept
- remove the column of 1's for the intercept
- recalculate Beta-hat and run an F-test to calculate
the overall adequacy of the model
''')

x=X[:,1:5]
xt=x.T
b_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(xt, x)),xt),y)

print('estimators for beta: ')
print(f'b1-hat: {b_hat[0]} \nb2-hat: {b_hat[1]} \nb3-hat: {b_hat[2]} \nb4-hat: {b_hat[3]} ')

dataset_no_int=aov_table(x,y,b_hat, 4,False)
dataset_no_int.summary()
```
