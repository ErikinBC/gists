import numpy as np
import pandas as pd
import plotnine
from plotnine import *

def loss_binomial(y,yhat):
    ll = -1*np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))
    return(ll)

def sigmoid(x):
    return( 1/(1+np.exp(-x)) )

from sklearn.linear_model import LogisticRegression, LinearRegression

# ---- Random circle data ---- #
np.random.seed(1234)
n_circ = 1000
pnoise = 3  # Number of continuous/binary noise features
X_circ = np.random.randn(n_circ,2+pnoise)
X_circ = X_circ + 1.0*np.random.randn(n_circ,1)
X_circ = np.hstack([X_circ,np.where(sigmoid(X_circ) > np.random.rand(n_circ,2+pnoise),1,0)])
y_circ = np.where(np.apply_along_axis(arr=X_circ[:,0:2],axis=1,func1d= lambda x: np.sqrt(np.sum(x**2)) ) > 1.2,1,0)

cn_type_circ = np.repeat('gaussian',X_circ.shape[1])

idx = np.arange(n_circ)
np.random.shuffle(idx)
idx_test = np.where((idx % 5) == 0)[0]
idx_train = np.where((idx % 5) != 0)[0]
# Plot the true distribution
df = pd.DataFrame({'y':y_circ,'var1':X_circ[:,0],'var2':X_circ[:,1]})

plotnine.options.figure_size = (4.0,3.5)
gg_circ = (ggplot(df,aes(x='var1',y='var2',color='y.astype(str)')) + theme_bw() + 
           geom_point(size=0.5) + scale_color_discrete(name='y') + 
           ggtitle('True distribution'))
gg_circ

rhomat = np.corrcoef(X_circ,rowvar=False)
rhomat = pd.DataFrame({'rho':rhomat[np.tril_indices_from(rhomat,-1)]})
gg_rho = (ggplot(rhomat, aes(x='rho')) + theme_bw() + 
          geom_histogram(bins=15, color='red',fill='grey') + 
          labs(y='Count',x='Correlation') + 
          ggtitle('Pairwise correlation between features'))
gg_rho

# Split data
X_train_circ = X_circ[idx_train]
X_test_circ = X_circ[idx_test]
y_train_circ = y_circ[idx_train]
y_test_circ = y_circ[idx_test]

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# Fit a non-linear machine learning model
clf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)
clf.fit(X_train_circ, y_train_circ)
acc_train = metrics.accuracy_score(y_train_circ, clf.predict(X_train_circ))
acc_test = metrics.accuracy_score(y_test_circ, clf.predict(X_test_circ))
print('Accuracy on training: %0.3f and test: %0.3f' % (acc_train, acc_test))

class hrt():
    def __init__(self, mdl):
        self.mdl = mdl  # some pretrained model with predict_probs
    def fit(self, X):
        self.p = X.shape[1]
        self.idx_bin = np.all((X==1) | (X == 0),0)  # Find binary features
        # Fit model to each column
        self.cmdls = {j: {'mdl':[], 'sig':0} for j in range(self.p)}
        for jj, isbin in enumerate(self.idx_bin):
            tmp_X, tmp_y = np.delete(X,jj,1), X[:,jj]
            if isbin:
                self.cmdls[jj]['mdl'] = LogisticRegression(penalty='none').fit(tmp_X, tmp_y)
            else:
                self.cmdls[jj]['mdl'] = LinearRegression().fit(tmp_X, tmp_y)
                tmp_res = tmp_y - self.cmdls[jj]['mdl'].predict(tmp_X)
                self.cmdls[jj]['sig'] = np.std(tmp_res)
                
    def pvals(self, Xtest, ytest, nsim=250):
        # Get the baseline score
        ll_test = loss_binomial(ytest, self.mdl.predict_proba(Xtest)[:,1])
        print(ll_test)
        for jj, isbin in enumerate(self.idx_bin):
            print('Column %i of %i' % (jj+1, len(self.idx_bin)))
            X_copy = Xtest.copy()
            tmp_X, tmp_y = np.delete(X_copy,jj,1), X_copy[:,jj]
            if isbin:
                phat = self.cmdls[jj]['mdl'].predict_proba(tmp_X)[:,1]
                ll_hold = np.zeros(nsim)
                for ii in range(nsim):
                    if (ii + 1) % 50 == 0:
                        print('simulation %i of %i' % (ii+1, nsim))
                    yrep = np.random.binomial(1,phat,len(phat))
                    X_copy[:,jj] = yrep
                    ll_hold[ii] = loss_binomial(ytest, self.mdl.predict_proba(X_copy)[:,1])
            else:
                yhat = self.cmdls[jj]['mdl'].predict(tmp_X)
                sig = self.cmdls[jj]['sig']
                ll_hold = np.zeros(nsim)
                for ii in range(nsim):
                    if (ii + 1) % 50 == 0:
                        print('simulation %i of %i' % (ii+1, nsim))
                    yrep = yhat + np.random.randn(len(yhat))*sig
                    X_copy[:,jj] = yrep
                    ll_hold[ii] = loss_binomial(ytest, self.mdl.predict_proba(X_copy)[:,1])
            pval = (np.sum(ll_hold < ll_test) + 1) / (nsim + 1)
            print('p-value: %0.3f' % pval)
            self.cmdls[jj]['pval'] = pval
        
inference = hrt(mdl=clf)
inference.fit(X_train_circ)
inference.pvals(X_test_circ, y_test_circ, 100)

res = pd.DataFrame({'pval':[inference.cmdls[m]['pval'] for m in inference.cmdls]})
print(res.reset_index().assign(is_noise = lambda x: np.where(x.index <= 1, False, True)))
