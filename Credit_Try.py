
import numpy as np
import AutoDataCleaner.AutoDataCleaner as adc
from AutoDataCleaner.AutoDataCleaner import read_csv
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

df =read_csv('train_t_credit.csv')
print('----------------SHOWINGTHE DATA SET -----------------')

data=adc.clean_me(df, 
            detect_binary=True,
            numeric_dtype=True,
            decision_tree=True, 
            one_hot=True,
            normalize=True,
            datetime_columns=[],
            remove_columns=[],
            high_corr_elimination=True,
            low_var_elimination=False,
            measuring_variable='A16',
            variable_to_encode=[],
            verbose=True)



y_train=data['A16']
X_train=data.drop(columns=['A16'])
df_test=read_csv('test_t_credit.csv')
y_test=data['A16']
X_test=data.drop(columns=['A16'])

model_LR= LogisticRegression()
model_LR.fit(X_train,y_train)
y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
acc_log=model_LR.score(X_train,y_train)
model_LR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
print(auc_roc)
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



