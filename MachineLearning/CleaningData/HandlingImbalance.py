import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pip._internal import models

df = pd.read_csv('balance-scale.data.txt',
                 names=['balance', 'var1', 'var2', 'var3', 'var4'])
# print(df.head())

# print(df['balance'].value_counts())
df['balance'] = [1 if b == 'B' else 0 for b in df.balance]

# print(df['balance'].value_counts())

y = df.balance
X = df.drop('balance', axis=1)

# print(X)
# Train model
clf_0 = LogisticRegression().fit(X, y)
# print(clf_0)
# Predict on training set
pred_y_0 = clf_0.predict(X)
# print(pred_y_0)

# print(accuracy_score(pred_y_0, y))
###############################################################################
'''
Up-sample Minority Class
'''
df_majority = df[df.balance == 0]
df_minority = df[df.balance == 1]

resampledminority = resample(
    df_minority, replace=True, n_samples=576, random_state=123)
# print(resampledminority.count())

df_upsampled = pd.concat([df_majority, resampledminority])
# print(df_upsampled.balance.value_counts())


y = df_upsampled.balance
X = df_upsampled.drop('balance', axis=1)

clf_1 = LogisticRegression().fit(X, y)
pred_y_1 = clf_1.predict(X)
print(accuracy_score(y, pred_y_1))
########################################################################


'''
 Down-sample Majority Class
 '''

df_majority = df[df.balance == 0]
df_minority = df[df.balance == 1]

resampledmajority = resample(
    df_majority, replace=False, n_samples=49, random_state=123)

df_downsampled = pd.concat([resampledmajority, df_minority])
# print(df_downsampled)

y = df_downsampled.balance
X = df_downsampled.drop('balance', axis=1)

clf_2 = LogisticRegression().fit(X, y)
pred_y_2 = clf_2.predict(X)

print(accuracy_score(y, pred_y_2))

##########################################################################

'''
Change The Performance Metric (AUROC)
'''

prob_y_2 = clf_2.predict_proba(X)

# Keep only the positive class
prob_y_2 = [p[1] for p in prob_y_2]

print(roc_auc_score(y, prob_y_2))


# imbalanced dataset
prob_y_0 = clf_0.predict_proba(X)
prob_y_0 = [p[0] for p in prob_y_0]

print(roc_auc_score(y, prob_y_0))

###########################################################################

'''
Using Penalised SVM algorithm
'''

y = df.balance
X = df.drop('balance', axis=1)

clf_3 = SVC(kernel='linear',
            class_weight='balanced',  # penalize
            probability=True)

clf_3.fit(X, y)

pred_y_3 = clf_3.predict(X)

print(accuracy_score(y, pred_y_3))

###############################################################################

'''
Using Tree based models
'''

y = df.balance
X = df.drop('balance', axis=1)

clf_4 = RandomForestClassifier()
clf_4.fit(X, y)

pred_y_4 = clf_4.predict(X)

print(accuracy_score(y, pred_y_4))
