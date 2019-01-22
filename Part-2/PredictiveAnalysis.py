# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 2
### GROUP 4
### Yipin, April, Kendra, Ratnadeep

import sys
import numpy as np
import pandas as pd
import warnings
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
from sklearn import svm, datasets
from sklearn import metrics
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

        
def ClassifyPred(myData, testsize = 0.20, nseed = 7): 
    valueArray = myData.values 
    # identifying the number of columns in the data
    numCol = len(myData.columns) -1
    X = valueArray[:,0:numCol]
    X = preprocessing.normalize(X, norm= 'l2')
    # separate out the predictor variable to build our predictive models
    Y = valueArray[:,numCol]
    test_size = testsize
    seed = nseed
    # spits out 4 outputs
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)  
    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    num_instances = len(X_train)
    seed = 3
    scoring = 'accuracy'
    
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVC', SVC()))
    models.append(('RF', RandomForestClassifier()))

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    results = []
    names = []
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    # Make predictions on validation dataset
    # iterate through the different models to validate the prediction values

    for model in models:
        mod = model[1]
        mod.fit(X_train, Y_train)
        predictions = mod.predict(X_validate)
        #print out the name of different classifier models
        text = "Validation check for : " + str(model[0])
        print(text)
        print(accuracy_score(Y_validate, predictions))
        print(confusion_matrix(Y_validate, predictions))
        print(classification_report(Y_validate, predictions))
    return(X_train, Y_train, X_validate, Y_validate, models)


def plotROC(X_train, Y_train, X_validate, Y_validate, models):
    classifyNames = ['KNN','CART', 'NB', 'RF']
    fig = plt.figure(figsize=(15,8))
    color = ['darkorange', 'purple', 'green', 'yellow', 'red']
    for i in range(len(models)):
        mod = models[i][1]
        name = str(models[i][0])
        if name in classifyNames:
            score = mod.fit(X_train, Y_train).predict_proba(X_validate)[:,1]
        else:
            score = mod.fit(X_train, Y_train).decision_function(X_validate)
        fpr, tpr, thresholds = roc_curve(Y_validate, score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,color = color[i] ,lw=2, label = str(name) + " AUC = %2f" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig('ROC.jpg')
    

def perform_ANOVA(df):
    #compare means by region
    df.groupby('region')['review_rating'].mean()

    d = {}
    for region in df['region'].unique():
        d[region] = df['review_rating'][df['region'] == region]

    test_result = stats.f_oneway(*d.values())

    #print results
    print("\n\nANOVA test for significant difference for review ratings by region." + 
          "\npvalue is: " + str(test_result.pvalue) +
          "\nReject null hypothesis and determine means are significantly different\n")
    

def do_regression(all_data):
    #Select the variables
    df_regression=all_data[all_data['PRCP']>0]
    df_regression=df_regression.reset_index()

    #Determine X, which have two attributes: preciptation and average temperature
    x=df_regression[['PRCP','TAVG']]
    x=x.values
    #Determine Y
    y=df_regression[['sentiment_polarity']]
    y=y.values
    
    #split data into training and testing set
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(x, y, 
                                        test_size=test_size, random_state=seed)
    
    # create linear regression
    regression = linear_model.LinearRegression()
    
    # train the model use the training set
    regression.fit(X_train, Y_train)
    # Make predictions using the testing set
    Y_predict = regression.predict(X_validate)
    # Evaluate the performance
    coef=regression.coef_
    mse=mean_squared_error(Y_validate, Y_predict)
    r2=r2_score(Y_validate, Y_predict)
    
    #print evaluation
    print('Do linear regression to all states.')
    print('Coefficient =',coef)
    print('Mean squared error =',mse)
    print('R square score =',r2)
    

    #plot 3D figure
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0],x[:,1],y,c='blue', marker='o', alpha=0.5)
    
    ax.plot_surface(np.array([[0, 0], [4000, 4000]]),
                    np.array([[0, 40], [0, 40]]),
                    regression.predict(np.array([[0, 0, 4000, 4000],
                                         [0, 40, 0, 40]]).T).reshape((2, 2))
                                          ,color='red' , alpha=0.7)
    ax.set_xlabel('Precipitation')
    ax.set_ylabel('Average Temperature')
    ax.set_zlabel('Polarity')
    plt.title('Regression of precipitation, average temperature, and polarity')   
    plt.savefig('Regression of precipitation, average temperature, and polarity'+ '.jpg')
    plt.show()
    
    
    #select five states
    states=['CA','VA','NE','NY','AL']#list(pd.unique(df_regression['state']))
    #Create a loop to get the results
    for i in states:
        #Determine X, which have two attributes: preciptation and average temperature
        x=df_regression[['PRCP','TAVG','state']]
        x=x[x['state']==i]
        del x['state']
        x=x.values
        #Determine Y
        y=df_regression[['sentiment_polarity','state']]
        y=y[y['state']==i]
        del y['state']
        y=y.values
    
        #split data into training and testing set
        test_size = 0.20
        seed = 7
        X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(x, y, 
                                        test_size=test_size, random_state=seed)
        # create linear regression
        regression = linear_model.LinearRegression()
    
        # train the model use the training set
        regression.fit(X_train, Y_train)
        # Make predictions using the testing set
        Y_predict = regression.predict(X_validate)

        # Evaluate the performance
        coef=regression.coef_
        mse=mean_squared_error(Y_validate, Y_predict)
        r2=r2_score(Y_validate, Y_predict)
    
        #print evaluation
        print('The state is',i)
        print('Coefficient =',coef)
        print('Mean squared error =',mse)
        print('R square score =',r2)


def main(args):
    warnings.simplefilter('ignore', UserWarning)
    df = pd.read_csv('merged_data.csv', encoding = 'Latin1')
    
    #creating a class attribute for ratings
    names = [0,1]
    ratingBin = [0,3,5]
    df['class_rating'] = pd.cut(df['review_rating'], ratingBin, labels = names)

    # shows a matrix of frequency between review ratings and the rating classifier
    pd.crosstab(df['review_rating'], df['class_rating'])

    #subsetting the dataframe to consider temperature, precipitation, polarity and region in predicting ratings
    #Can we use these attributes to predict the overall yelp review? 
    # positive review : review ratings 4-5
    # negative review : review ratings 1-3
    df2 = df[['TAVG', 'PRCP', 'region_idx', 'class_rating']]
    df2['class_rating']= df2['class_rating'].astype(np.float64)
    
    #run the classifiers
    X_train, Y_train, X_validate, Y_validate, models = ClassifyPred(df2)
    plotROC(X_train, Y_train, X_validate, Y_validate, models)    

    perform_ANOVA(df)
    do_regression(df)
    

if __name__ == "__main__":
    main(sys.argv)