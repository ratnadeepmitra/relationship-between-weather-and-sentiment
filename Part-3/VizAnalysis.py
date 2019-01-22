# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 2
### GROUP 4
### Yipin, April, Kendra, Ratnadeep

from datetime import datetime, date
from geopy.distance import vincenty
from itertools import combinations
from itertools import cycle
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist 
from nltk.tokenize import RegexpTokenizer
from pandas.plotting import scatter_matrix
from pandas.tools.plotting import scatter_matrix
from plotly.graph_objs import *
from plotly.graph_objs import Surface
from pprint import pprint
from scipy import interp
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import ward, dendrogram, linkage
#from sklearn import cross_validation
#from sklearn.cross_validation import train_test_split
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm, datasets
from sklearn import model_selection
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statistics import mode
from textblob import TextBlob
from wordcloud import WordCloud
import community
from community import community_louvain
import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff  
import plotly.plotly as py
import pylab as pl
import re
import sys
import warnings


import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


def main(args):
    warnings.simplefilter('ignore', UserWarning)
    
    api_username = input("Enter Plotly API Username: ")
    api_key = input("\nEnter Plotly API Key:")

    plotly.tools.set_credentials_file(username=api_username, api_key=api_key)

    df = pd.read_csv('merged_data.csv', encoding = 'latin1')

    do_kmeans_plotly(df)
    do_dbscan_plotly(df)
    
    do_topicmodeling(df)
    do_extra_analysis(df)
    
    do_networkAnalysis(df)
    
    do__HCluster_ROC(df)

def do_kmeans_plotly(all_data):
    all_data = pd.read_csv('merged_data.csv', encoding = 'latin1')
    dt = pd.concat([all_data['rest_lat'], all_data['rest_lon'], all_data['TAVG'], all_data['sentiment_polarity']], 
                   axis = 1, keys = ['rest_lat', 'rest_lon', 'TAVG', 'sentiment_polarity'])
    x = dt.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    k = 3
    kmeans = KMeans(n_clusters = k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    ## Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    ## Convert our high dimensional data to 2 dimensions using PCA
    pca2D = decomposition.PCA(2)
    ## Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    pcadf = pd.DataFrame(plot_columns)
    pcadf.columns = ['X','Y']
    
    trace =  go.Scatter(
            x = pcadf['X'],
            y = pcadf['Y'],
            mode = 'markers',
            marker=dict(color=kmeans.labels_,
            size = 7.5,
            line = dict(width=0.3))
            )
    
    myData = [trace]
    
    ## Add title
    myLayout = go.Layout(
            title = "K-Means Clustering  (Number of clusters: 3, Silhouette score: "+ str(round(silhouette_avg,3))+")"
            )
    
    ## Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)

    ## Create the scatterplot
    py.iplot(myFigure, filename='kmeans')
    py.plot(myFigure, filename='kmeans')
    
    
def do_dbscan_plotly(df):
    dt = pd.concat([df['rest_lat'], df['rest_lon'], df['TAVG'], df['sentiment_polarity']], 
                   axis = 1, keys = ['rest_lat', 'rest_lon', 'TAVG', 'sentiment_polarity'])
    x = dt.values
    ## Normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)

    i = 0.4
    j = 30
    db = DBSCAN(eps=i, min_samples=j).fit(normalizedDataFrame) 
    db_labels = db.labels_

    ## Calculate the sihouette score
    db_sihouette_score = metrics.silhouette_score(x, db_labels)
    
    pca2D = decomposition.PCA(2)
    ## Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    pcadf = pd.DataFrame(plot_columns)
    pcadf.columns = ['X','Y']
    
    trace =  go.Scatter(
            x = pcadf['X'],
            y = pcadf['Y'],
            mode = 'markers',
            marker=dict(color=db.labels_,
            size = 7.5,
            line = dict(width=0.3))
            )
    
    myData = [trace]
    
    ## Add title
    myLayout = go.Layout(
            title = "DBSCAN  (EPS: 0.4, Silhouette score: "+ str(round(db_sihouette_score,3))+")"
            )
    
    ## Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)

    ## Create the scatterplot
    py.iplot(myFigure, filename='dbscan')
    py.plot(myFigure, filename='dbscan')


#define function for running LDA and spitting out topics
def topic_model(docs):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    tf = tf_vectorizer.fit_transform(docs)
    tf_feature_names = np.array(tf_vectorizer.get_feature_names())

    #run LDA
    lda = LatentDirichletAllocation(
            n_topics=3,
            learning_method='batch', 
            random_state=0,
            verbose=True
        ).fit(tf)

    #display results
    k = 20
    for topic_idx, topic in enumerate(lda.components_):
        print('\nTopic: ', topic_idx)
        print(' '.join(tf_feature_names[topic.argsort()[::-1][:k]]))


def do_topicmodeling(df):
    #create list from text of each yelp review
    my_docs = list(df['text'])
    
    #tokenize and vectorize
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=100, stop_words='english')
    tf = tf_vectorizer.fit_transform(my_docs)
    tf_feature_names = np.array(tf_vectorizer.get_feature_names())
    
    #run LDA
    lda = LatentDirichletAllocation(
            n_topics=10,
            learning_method='batch', 
            random_state=0,
            verbose=True
        ).fit(tf)
    
    #display results
    k = 20
    for topic_idx, topic in enumerate(lda.components_):
        print('\nTopic: ', topic_idx)
        print(' '.join(tf_feature_names[topic.argsort()[::-1][:k]]))
    
    ###--- topic modelling by region
    #make list to contain all reviews by region (each document is a region)
    my_docs2 = []

    #for each region, make a set of documents
    region = df['region'].unique()

    docs_0 = list(df['text'][df['region']=='Pacific'])
    docs_1 = list(df['text'][df['region']=='NorthCentral'])
    docs_2 = list(df['text'][df['region']=='SouthCentral'])
    docs_3 = list(df['text'][df['region']=='Southeast'])
    docs_4 = list(df['text'][df['region']=='MidAtlantic'])
    docs_5 = list(df['text'][df['region']=='Northwest'])
    docs_6 = list(df['text'][df['region']=='NewEngland'])
    docs_7 = list(df['text'][df['region']=='Mountain'])
    docs_8 = list(df['text'][df['region']=='Midwest'])
    
    topic_model(docs_0)
    topic_model(docs_1)
    topic_model(docs_2)
    topic_model(docs_3)
    topic_model(docs_6)
    topic_model(docs_7)
    topic_model(docs_8)


def do_extra_analysis(df):
    ###--- weather words frequency ---###

    #import list of weather related words
    #via: http://www.enchantedlearning.com/wordlist/weather.shtml
    weatherwords = ['balmy', 'blustery', 'breeze','breezy','cloud', 'cloudy', 'clouds', 
                'dew point', 'fog', 'foggy', 'freezing rain', 
                'hail', 'heat index','humid', 'humidity', 'lightening', 
                'overcast', 'partly cloudy', 'precipitation', 
                'rain', 'rainy', 'raining', 'rained', 'sleet', 'sleeting', 
                'smog', 'snow', 'snowing', 'snowed',  'snowfall', 'snowstorm', 
                'snow shower', 'sunrise', 'sunset', 'thunder', 'thunderstorm', 
                'tornado', 'hurricane', 'weather', 'wind chill', 'windy']
   
    #tokenize and check for words in weather list
    df['tokens'] = ''
    stopWords = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(df.index)):
        clean_tokens = []    
        words = tokenizer.tokenize(df.loc[i,'text'])
        for word in words:
            word = word.lower()
            if word not in stopWords:
                if word.isalpha(): # Drop all non-words
                    if word in weatherwords:
                        clean_tokens.append(word)
    
        df.loc[i, 'tokens'] = ' '.join(clean_tokens)

    ll = []
    for sList in df['tokens']:
        ll = ll + sList.split(' ')

    #compute frequency  
    fdist = FreqDist(ll)

    #compute % of reviews that include a weather word
    ((df['tokens'].apply(len) > 0).sum())/(len(df))*100

    #display reviews that mention weather
    for t in df[df.tokens.apply(len) > 0]['text']:
        print(t, end='\n\n')


    ###--- sentiment analysis reviews ---###
    df.head()

    #low rating, positive sentiment
    tmp = df[(df['sentiment_idx'] == 1.0) & (df['review_rating'] == 1.0)]
    tmp = tmp.sort_values('sentiment_polarity')
    for t in tmp['text']:
        print(t, end='\n\n', file=open('text_lowrtg_positive.txt-2', 'a'))

    #low rating, negative sentiment
    for t in df['text'][(df['sentiment_idx'] == -1.0) & (df['review_rating'] == 1.0)]:
        print(t, end='\n\n', file=open('text_lowrtg_negative.txt', 'a'))

    #high rating, positive sentiment
    for t in df['text'][(df['sentiment_idx'] == 1.0) & (df['review_rating'] == 5.0)]:
        print(t, end='\n\n', file=open('text_highrtg_positive.txt', 'a'))

    #high rating, negative sentiment
    for t in df['text'][(df['sentiment_idx'] == -1.0) & (df['review_rating'] == 5.0)]:
        print(t, end='\n\n', file=open('text_highrtg_negative.txt', 'a'))


    ###-- sentiment by region --###

    df['sentiment_polarity'] = df['sentiment_polarity'].astype(float)
    df.groupby('region')['sentiment_polarity'].mean()

    df.groupby(['region', 'sentiment_polarity']).mean()


def networkFoodData(df):
    ## tokenize the review text and store it in a new column
    food = wn.synset('food.n.02')
    foods = list(set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
    ff = []
    for item in foods:
        item = item.lower() 
        # exluding the words 'delicious' and 'gem' - anomaly in the dataset
        if item not in ('delicious','gem'):
            ff.append(item)
        
    ## sort the list of foods
    ff.sort()
    
    df['tokens'] = ''
    df = df.astype(object)
    stopWords = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(df.index)):
        clean_tokens = []    
        words = tokenizer.tokenize(df.loc[i,'text'])
        for word in words:
            word = word.lower()
            if word not in stopWords:
                if word.isalpha(): # Drop all non-words
                    if word in ff:
                        clean_tokens.append(word)
        df.loc[i,'tokens'] = clean_tokens

    ll = []
    for sList in df['tokens']:
        for list_item in sList:
            ll.append(list_item)
        

    fdist = FreqDist(ll)
    
    ## Picked the 15 highest frequency words
    voca = pd.DataFrame(fdist.most_common(15))

    voca.columns = ('word', 'counts')
    voca_pivot = voca.set_index('word').T

    ## create an assotiation table of the food items and their occurence
    for i in range(len(voca.index)):
        foodnm = voca.loc[i,'word']
        for j in range(len(df['tokens'])):
            if foodnm in df.loc[j,'tokens']:
                voca_pivot.loc[j, foodnm] = 1
            else:
                voca_pivot.loc[j, foodnm] = 0
   
    voca_pivot = voca_pivot.fillna(0)
    voca_pivot = voca_pivot.astype(int)    
    voca_pivot = voca_pivot.drop(voca_pivot.index[0])    
    voca_pivot = voca_pivot.loc[:, ~voca_pivot.columns.str.contains('word')]
    voca_pivot = voca_pivot.reset_index(drop = True)

    ## function to calcuate support
    ## mostly the same as other except i joined with space rather than comma
    def get_support(df, sample_space):
        pp = []
        for cols in combinations(df, 2):
                s = df[list(cols)].all(axis=1).sum()
                pp.append([" ".join(cols), s])
        sdf = pd.DataFrame(pp, columns=['Pattern', 'Count'])
        return sdf


    sample_space = len(df)
    s = get_support(voca_pivot, sample_space)
    
    ## subsetting the data to only include items that had more than 2 counts of associations
    new_s = s[s.Count > 2]
    
    #tkplot 
    index0 = new_s.Pattern.str.split('\s+').str[0]
    #has county name
    index1 = new_s.Pattern.str.split('\s+').str[1]
    #new1 = index1.replace("County", "")
    #new_s['Pattern'] = index0.str.strip() + ' ' + index1.str.strip()
    new_s['f1'] = index0.str.strip() 
    new_s['f2'] = index1.str.strip()
    del new_s['Pattern']
    new_s = new_s[['f1', 'f2', 'Count']]
    new_s = new_s.sort_values(['Count'], ascending = 0)
    new_s.to_csv('NetworkFood.txt', header=None, index=None, sep=',', mode='w', quoting = csv.QUOTE_NONE, escapechar=',')
    
    FILE1=open("NetworkFood.txt", "rb")
    G=nx.read_edgelist(FILE1, delimiter=",",create_using=nx.Graph(), nodetype=str,data=[("weight", int)])
    FILE1.close()
    print("G is:" ,G.edges(data=True), "\n\n\n\n")
    edge_labels = dict( ((u, v), d["weight"]) for u, v, d in G.edges(data=True) )
    pos = nx.random_layout(G)
    nx.draw(G, pos, edge_labels=edge_labels, with_labels=True)
    #changing the node color to blue
    nx.draw_networkx_nodes(G, pos, node_color = 'r')
    #labeling the edges of the network with the weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
    plt.show()
    NetworkAnalysis(G, filename = 'lowRatingResults.txt')
    plotNetworkFood(G, pos)
    

def plotNetworkFood(G, pos):

    dmin=1
    ncenter=0
    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d

    p=nx.single_source_shortest_path_length(G,ncenter)

    edge_trace = Scatter(
            x=[],
            y=[],
            line=Line(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]].tolist()
        x1, y1 = pos[edge[1]].tolist()
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=Marker(
                    showscale=True,
                    colorscale='YIGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    colorbar=dict(
                            thickness=15,
                            title='Node Connections',
                            xanchor='left',
                            titleside='right'
                            ),
                            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node].tolist()
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        
    g_nodes = G.nodes()
    for node, adjacencies in enumerate(G.adjacency_list()):
        node_trace['marker']['color'].append(len(adjacencies))
        node_info = '# of connections: '+str(len(adjacencies)) + ' for ' + g_nodes[node]
        node_trace['text'].append(node_info)
        

    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                         title='<br>Network graph of food items in Yelp reviews',
                         titlefont=dict(size=16),
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    
    py.plot(fig, filename='foodNetwork')

def NetworkAnalysis(G, filename = 'highRatingResults.txt'):
#standard metrics local metrics
    nbr_nodes = nx.number_of_nodes(G)
    nbr_edges = nx.number_of_edges(G)
    nbr_components = nx.number_connected_components(G)
    F = open(filename, 'w')

#    t1 = "Number of nodes:" + str(nbr_nodes) 
#    t2 = "Number of edges:" + str(nbr_edges) 
#    t3 = "Number of connected components:" + str(nbr_components) 
    F.write("Number of nodes:" + str(nbr_nodes) + "\n")
    F.write("Number of edges:" + str(nbr_edges) + "\n")
    F.write("Number of connected components:" + str(nbr_components) + "\n")
   
   # F.close()

    #betweeness
    betweenList = nx.betweenness_centrality(G)
    #print("The list of betweenness centrality is", str(betweenList), "\n")
    F.write("The list of betweenness centrality is" + str(betweenList) + "\n")
    #all the items have less than 1 betweenness centrality which indicate that there is no 
    #item that lie inbetween the connection between two items. 

    #degree
    degreeCentrality = nx.degree_centrality(G)
    F.write("The degrees of centrality is " + str(degreeCentrality) + "\n")

    #clustering coefficient
    #clustering coefficient for each nodes
    F.write("The clustering coefficients are " + str(nx.clustering(G)) + "\n")


    partition = community_louvain.best_partition(G)
    F.write("The community modularity is " + str(community_louvain.modularity(partition, G)) + "\n")
    #which suggest that there isn't a strong community

    #global network metrics (metric to explain whole network not just a part)
    #diameter - the max of shortest distances between nodes
    F.write("The diameter is " + str(nx.diameter(G)) + "\n")

    #density
    F.write("The density is " + str(nx.density(G)) + "\n")
    #not particularly low nor high in density

    #triangles
    F.write("The triangle is " + str(nx.triangles(G)) + "\n")


    #average clustering coefficient for the graph
    avgclu = nx.average_clustering(G)
    F.write("The average clustering is " + str(avgclu) + "\n")
    #average degree centrality
    tot = []
    for food in degreeCentrality:
        item = degreeCentrality[food]
        tot.append(item)
        avgdeg= np.average(tot)   
    F.write("The average degree centrality is " + str(avgdeg) + "\n") 
    
    #average betweenness centrality
    l = []
    for f in betweenList:
        item = betweenList[f]
        l.append(item)
        avgB = np.average(l)   
    F.write("The average betweenness centrality is " + str(avgB) + "\n") 
    F.close()


def do_networkAnalysis(df):
    ## keep only 1-star to 3-star for low ratings
    df1 = df[df['review_rating'] < 4]
    df1 = df1.reset_index(drop = True)
    networkFoodData(df1)
#    NetworkAnalysis(G, filename = 'lowRatingResults.txt')
#    plotNetworkFood(G, username = 'ahc72', api_key='hKxM7sdoOGLRIv6EhE34')   
    ## keep only 5-star ratings
    df2 = df[df['review_rating'] == 5]
    df2 = df2.reset_index(drop = True)   
    networkFoodData(df2)
#    NetworkAnalysis(G, filename = 'highRatingResults.txt')
#    plotNetworkFood(G, username = 'ahc72', api_key='hKxM7sdoOGLRIv6EhE34')


def hierarchical_plotly(df):
    
    #initializing the text array to contain all reviews by region
    text_array = []
    #getting the list of the regions
    region = df['region'].unique()
    for reg in range(9):
        data = np.array(df.loc[df.region==region[reg], ['text']])
        b = data.ravel()
        data_str = ' '.join(b)
        text_array.append(data_str)

    txt = np.array(text_array)

    vect = CountVectorizer()
    bag = vect.fit_transform(txt)
    #checking the vocab
    #print(vect.vocabulary_)
    S = 1-cosine_similarity(bag)
    lnk = ward(S)
    #evaluate the quality of ward cluster using silhouette scores
    ward_label = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
    w_label = ward_label.fit_predict(lnk)   
    h_silhouette_score = silhouette_score(lnk, w_label)
    #use plotly to create a dendrogram
    fig = ff.create_dendrogram(
        S, orientation='left', labels=region,
       linkagefun=lambda x: linkage(S, 'ward', metric='euclidean')
    )
    #Update the width, height, and title
    fig['layout'].update(width = 800, height = 600, 
          title= 'Hierarchical Clustering Dendrogram of Reviews by Region (sihouette score = ' 
          + str(round(h_silhouette_score, 2)) + ')')
    fig['layout'].update(xaxis=dict(#range=[0, 0.05],
                                  title='cosine similarity distance')
                         )
    py.plot(fig, filename='dendrogram_with_labels')


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
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)  
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
        kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
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
    

def roc_plotly(X_train, Y_train, X_validate, Y_validate, models):
    classifyNames = ['KNN','CART', 'NB', 'RF']
    fig = plt.figure(figsize=(15,8))
    #define colors
    color = ['darkorange', 'purple', 'green', 'yellow', 'red']
    #use data to record traces
    data = []    
    lw = 2
    #calculate trace for each model
    for i in range(len(models)):
        mod = models[i][1]
        name = str(models[i][0])
        if name in classifyNames:
            score = mod.fit(X_train, Y_train).predict_proba(X_validate)[:,1]
        else:
            score = mod.fit(X_train, Y_train).decision_function(X_validate)
        fpr, tpr, thresholds = roc_curve(Y_validate, score)
        roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr,color = color[i] ,lw=2, label = str(name) + " AUC = %2f" % roc_auc)
        
        trace1 = go.Scatter(x=fpr, y=tpr, 
                    mode='lines', 
                    line=dict(color = color[i], width=lw),
                    name=str(name) + " AUC = %2f" % roc_auc)
        data.append(trace1)
    
    # trace2 represents the line of rance chance
    trace2 = go.Scatter(x=[0, 1], y=[0, 1], 
                mode='lines', 
                line=dict(color='navy', width=lw, dash='dash'),
                name='random chance')
    data.append(trace2)
    
    layout = go.Layout(title='Receiver operating characteristic',
                   xaxis=dict(title='False Positive Rate'),
                   yaxis=dict(title='True Positive Rate'))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='ROC Curve')


def do__HCluster_ROC(df):
    names = [0,1]
    ratingBin = [0,3,5]
    df['class_rating'] = pd.cut(df['review_rating'], ratingBin, labels = names)    
    hierarchical_plotly(df)
    # shows a matrix of frequency between review ratings and the rating classifier
    pd.crosstab(df['review_rating'], df['class_rating'])
    df2 = df[['TAVG', 'PRCP', 'region_idx', 'class_rating']]
    df2['class_rating']= df2['class_rating'].astype(np.float64)
    #hierarchical_plotly(df)
    #run the classifiers
    X_train, Y_train, X_validate, Y_validate, models = ClassifyPred(df2)
    roc_plotly(X_train, Y_train, X_validate, Y_validate, models)





if __name__ == "__main__":
    ## Execute only if run as a script  
    main(sys.argv)