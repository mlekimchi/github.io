---
layout: post
title: I wrote code for an *inferior* ANOVA table and t-test in Python
published: true
comments: true
---

You should definitely use ```SciPy``` and ```statsmodels``` for statistical modeling in Python. For my linear regression class, I misheard that we had to do the variance analysis without any built-in or external packages...by hand! so I wrote an analysis of variance class ```aov_table()``` and a t-test function ```meanCI()``` to test the significance of a regression and significance of individual regressors. As the title says, my stuff is inferior to the standard packages lol. You can check out my code below or on [Kaggle](https://www.kaggle.com/code/emilykchang/ra-hw5-6)

### ```aov_table()``` Analysis of variance: overall model adequacy
The ```aov_table()``` class takes in your linear model and estimators and creates an ANOVA table object with your standard ANOVA table values. When you create your object, you'll get a readout of (almost) all associated info: values and degrees of freedom for the sums of squares, mean sums of squares, F-statistic, and R$^2$/R$^2_{adjust}$. ```aov_table()``` comes with class functions to pull out specific values from the table. You can also specify whether in include the intercept in the analysis.

### ```meanCI()``` Hypothesis testing with t-distribution: individual regressor significance
I also wrote the function ```meanCI()``` to test individual regressors (t-test) for your linear model. It uses values calculated with ```aov_table()``` and gives you the t-score, (1-$\alpha$)% CI, and hypothesis conclusion.

### Thoughts?
I used ```statsmodels``` to cross check my code. I much prefer ```statsmodels``` over my stuff because it is so much more readable! If I have time, maybe I can play around with the output format. I really enjoyed practicing Python, specifically building a class and working with ```fstrings```. I also gained a much deeper understanding of sums of squares which had always been nebulously floating around in my head.

## Example and Code
See my [Kaggle workbook](https://www.kaggle.com/code/emilykchang/ra-hw5-6) for the code and an example :)

### Code for analysis of variance object
```python 
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
```

### Code for testing significance of individual regressors

```python
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
    degreefreedom=len(X)-9-1
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
```
