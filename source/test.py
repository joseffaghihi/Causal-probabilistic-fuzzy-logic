import numpy as np
import pandas as pd
from source.utils import Score
from source.Fuzzy import Fuzzy
from sklearn.model_selection import train_test_split



df= pd.read_csv('https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv', header = None)

df.dataframeName = 'data'

cols =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [i for i in range(25)]

df.columns = cols
print(df.head())


#precising variables type
binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
contfeats = [i for i in range(25) if i not in binfeats]
perm = binfeats + contfeats
df = df.reset_index(drop=True)
df.head()

X = df[perm].values
treatment = df['treatment'].values
y = df['y_factual'].values
y_cf = df['y_cfactual'].values
tau = df.apply(lambda y: y['y_factual'] - y['y_cfactual'] if y['treatment']==1
               else y['y_cfactual'] - y['y_factual'],
               axis=1)
mu_0 = df['mu0'].values
mu_1 = df['mu1'].values

# train and test
itr, ite = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=1)
X_train, treatment_train, y_train, y_cf_train, tau_train, mu_0_train, mu_1_train = X[itr], treatment[itr], y[itr], y_cf[itr], tau[itr], mu_0[itr], mu_1[itr]
X_val, treatment_val, y_val, y_cf_val, tau_val, mu_0_val, mu_1_val = X[ite], treatment[ite], y[ite], y_cf[ite], tau[ite], mu_0[ite], mu_1[ite]

m=min(len(y_val),len(y_train))

test=Fuzzy()
new=test.fuzzify(df,[1,2],0,1)
eval=Score(y_train[:m],treatment_train[:m],y_cf_train[:m],mu_0_train[:m],mu_1_train[:m])
print('ATE ',eval.evaluate(y_val,y_cf_val)[1])
y = new['y_factual_low'].values
y_cf = new['y_cfactual_low'].values
y_val, y_cf_val=y[ite], y_cf[ite]
low=eval.evaluate(y_val,y_cf_val)[1]
print('ATE low ',low)
y = new['y_factual_average'].values
y_cf = new['y_cfactual_average'].values
y_val, y_cf_val=y[ite], y_cf[ite]
medium=eval.evaluate(y_val,y_cf_val)[1]
print('ATE medium ',medium)
y = new['y_factual_high'].values
y_cf = new['y_cfactual_high'].values
y_val, y_cf_val=y[ite], y_cf[ite]
high=eval.evaluate(y_val,y_cf_val)[1]
print('ATE high',high)
print('The average value of low, medium and high :', np.mean([low,medium,high]))






