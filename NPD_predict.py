import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

well_db= pd.read_csv('./data/wellbore_exploration_all_cut.csv',sep=';')


df_short=pd.DataFrame(well_db,columns= ['Block', 'Drilling_operator', 'Discovery','1st_level_HC','Entry_year']).astype(object).replace(np.nan, 'None')

print(df_short.dtypes)

#encode text data
def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = sk.preprocessing.LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)
    return df

df= Encoder(df_short)

df.drop(index=df[df['Entry_year'] == 'None'].index, inplace=True)


X = df[['Block', 'Drilling_operator','Entry_year','1st_level_HC']] #features
y = df['Discovery'] #labels

#create test and train set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)

#create classifier
clf = RandomForestClassifier(n_estimators=100, min_samples_split= 3, max_features= 'log2')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


#show confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()


#predict Block:16/4, Operator: Lundind, Target(1st_level_HC):Ty FM
prediction = clf.predict([['24','39',2021, '80']])
print ('Predicted Result: ', prediction)   # 0:Dry 2:Discovery

#show feature importances
featureImportances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(featureImportances)

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), featureImportances, align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
plt.tight_layout()
plt.show()


#sn.barplot(x=round(featureImportances,4), y=featureImportances)
#plt.xlabel('Features Importance')
#plt.show()'''
