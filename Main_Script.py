import time 
import numpy as np 
import AutoDataCleaner.AutoDataCleaner as adc
from AutoDataCleaner.AutoDataCleaner import read_csv
from sklearn import metrics
from sklearn.linear_model import LogisticRegression




ini=time.time()




print('----------------SHOWINGTHE DATA SET -----------------')
# Split into train and test sets
df =read_csv('train_t.csv')
print(df.info())
data=adc.clean_me(df, 
            detect_binary=True,
            numeric_dtype=True,
            decision_tree=True, 
            one_hot=True,
            normalize=True,
            datetime_columns=[],
            remove_columns=['Name'],
            high_corr_elimination=True,
            low_var_elimination=True,
            measuring_variable='Survived',
            variable_to_encode=[],
            outlier_removal=True,
            duplicated_var=True,
            duplicated_rows_remove=True,
            variables_uni=True,
            clean=True,
            verbose=True)

print(data)
fin=time.time()
tiempo_ejecucion=fin-ini
print(tiempo_ejecucion)


y_train=data['Survived']
X_train=data.drop(columns=['Survived'])
df_test=read_csv('test_t.csv')
y_test=data['Survived']
X_test=data.drop(columns=['Survived'])

X=data
y=data['Survived']
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


