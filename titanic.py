# Import libraries
import sys
import scipy
import numpy as np
import matplotlib
import pandas as pd
import sklearn
import csv


# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# Load dataset
location = "Data/train.csv"
dataset = read_csv(location)

# process data
def data_preprocessing(dataset):
    # functions
    def substrings_in_string(big_string,substrings):
        for substring in substrings:
            if big_string.find(substring)!=-1:
                return substring
        return np.nan

    def term_hash(df,Original_Feature,term_list):
        count = 0
        l = len(term_list)
        for term in term_list:
            newcol = Original_Feature+'_'+term
            hash_array = np.zeros(l)
            hash_array[count]=1
            df[newcol] = df[Original_Feature].replace(term_list,hash_array)
            count+=1

    # Clean up data - Sex
    dataset['Sex'] = dataset['Sex'].replace('male',0)
    dataset['Sex'] = dataset['Sex'].replace('female',1)

    # Clean up data - Cabin to Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    dataset['Deck'] = dataset['Cabin'].map(lambda x: substrings_in_string(str(x),cabin_list))
    dataset['Deck'] = dataset['Deck'].replace(np.NaN,'Unknown')
    term_hash(dataset,'Deck',cabin_list)

    # Clean up data - Name to Title
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev','Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess','Don', 'Jonkheer']
    title_list_cond = ['Mrs', 'Mr', 'Master', 'Miss']

    def replace_titles(x):
        title = x['Title']
        if title in ['Don','Major','Jonkheer','Rev','Col','Capt']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    dataset['Title'] = dataset['Name'].map(lambda x: substrings_in_string(str(x),title_list))
    #dataset['Title'] = dataset.apply(replace_titles, axis=1)
    term_hash(dataset,'Title',title_list)

    # Clean up data - Add family size
    dataset['Family_Size'] = dataset['SibSp']+dataset['Parch']

    # Clean up data - Fare per person
    dataset['Fare_Per_Person']=dataset['Fare']/(dataset['Family_Size']+1)

    # Clean up data - Embarked
    embarked_list = ['C','Q','S']
    term_hash(dataset,'Embarked',embarked_list)

    # Remove columns 
    dataset = dataset.drop(['Name','Title','Ticket','Embarked','Cabin','Deck'],axis=1)

    #replace all Nan 
    dataset['Age'] = dataset['Age'].fillna(value = dataset['Age'].notna().mean(),inplace=True)
    dataset = dataset.replace(np.NaN,0)

    return dataset
dataset = data_preprocessing(dataset)

# Split out training and validation sets - validation data to be held abck until the very last stage of model validation
X = dataset.drop(['Survived'],axis=1).values
Y = dataset['Survived'].values
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=1)

# Build models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('MLP',MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)))

# Evaluate models
results = []
names = []
for name, model in models:
	skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# use best model on validation set
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# Load testing data
location = "Data/test.csv"
test_set = read_csv(location)
test_set = data_preprocessing(test_set)
test_set = test_set.values

# generate perdictions on testing data
final_perdictions = model.predict(test_set)

# write final perdctions to a csv for submission to kaggle

final_df = pd.DataFrame({'PassengerId':test_set[:,0],'Survived':final_perdictions})

final_df['PassengerId'] = final_df['PassengerId'].astype('int32')
final_df['Survived'] = final_df['Survived'].astype('int32')

final_df.to_csv(r'Data/Final_Predictions.csv',index=False)


