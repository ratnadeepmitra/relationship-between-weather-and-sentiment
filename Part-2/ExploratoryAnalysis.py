# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 2
### GROUP 4
### Yipin, April, Kendra, Ratnadeep

import sys
import os
import numpy as np
import pandas as pd
import warnings
from nltk.probability import FreqDist 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, date
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score
from geopy.distance import vincenty
from scipy import stats
from statistics import mode
from pandas.plotting import scatter_matrix
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.cluster.hierarchy import ward, dendrogram, linkage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import wordnet as wn
from itertools import combinations
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


def main(args):
    warnings.simplefilter('ignore', UserWarning)
    
    df = pd.read_csv('merged_data.csv', encoding = 'latin1')
    
    #df.to_csv(r'merged_data.csv', header = True, index = False)
    #print("Step 2")
    #print(df)
         
    #res = pd.read_csv('Restaurant_Details_CleanData.csv', encoding = 'latin1')
    df = create_bins_region(df)
    df.to_csv('merged_data.csv', header = True, index = False)
    #print("Step 3")
    #print(df)
    
    #weather = pd.read_csv('merged_data.csv', encoding = 'latin1')
    df = create_bins_weather(df)
    df.to_csv('merged_data.csv', header = True, index = False)
    #print("Step 4")
    #print(df)
    
    df = do_sentiment_analysis(df)
    df.to_csv('merged_data.csv', header = True, index = False)
    sentiment = ["positive","negative", "all"]
    for s in sentiment:
        draw_word_cloud(df, s)
    
    run_summary_stats(df)
    draw_histograms(df)
    draw_scatterplots(df)
    
    #all_data = pd.read_csv('merged_data.csv', encoding = 'latin1')
    hierarchCluster(df)
    do_kmeans(df)
    do_dbscan(df)
    
    do_apriori()
    

def create_bins_region(res):
    ## Divide the states into 10 regions (bins)
    Region1 = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']
    Region2 = ['NJ', 'NY', 'DE', 'DC', 'MD', 'PA', 'VA', 'WV']
    Region3 = ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN']
    Region4 = ['IL', 'IN', 'MI', 'MN', 'OH', 'WI']
    Region5 = ['AR', 'LA', 'NM', 'OK', 'TX']
    Region6 = ['IA', 'KS', 'MO', 'NE']
    Region7 = ['CO', 'MT', 'ND', 'SD', 'UT', 'WY']
    Region8 = ['AZ', 'CA', 'HI', 'NV']
    Region9 = ['AK', 'ID', 'OR', 'WA']

    ## Check the restaurants' locations and classify into appropriate bins
    for i in range(len(res.index)):
        if res.loc[i,'state'] in Region1:
            res.loc[i,'region'] = 'NewEngland'
            res.loc[i,'region_idx'] = 1
        elif res.loc[i,'state'] in Region2:
            res.loc[i,'region'] = 'MidAtlantic'
            res.loc[i,'region_idx'] = 2
        elif res.loc[i,'state'] in Region3:
            res.loc[i,'region'] = 'Southeast'
            res.loc[i,'region_idx'] = 3
        elif res.loc[i,'state'] in Region4:
            res.loc[i,'region'] = 'NorthCentral'
            res.loc[i,'region_idx'] = 4
        elif res.loc[i,'state'] in Region5:
            res.loc[i,'region'] = 'SouthCentral'
            res.loc[i,'region_idx'] = 5
        elif res.loc[i,'state'] in Region6:
            res.loc[i,'region'] = 'Midwest'
            res.loc[i,'region_idx'] = 6
        elif res.loc[i,'state'] in Region7:
            res.loc[i,'region'] = 'Mountain'
            res.loc[i,'region_idx'] = 7
        elif res.loc[i,'state'] in Region8:
            res.loc[i,'region'] = 'Pacific'
            res.loc[i,'region_idx'] = 8
        elif res.loc[i,'state'] in Region9:
            res.loc[i,'region'] = 'Northwest'
            res.loc[i,'region_idx'] = 9
      
    res['region'].value_counts()    
    res = res[~res['region'].isnull()]
    res = res.reset_index(drop=True)
    return(res)
            

def create_bins_weather(weather):
    formatter_string = "%Y-%m-%d"
    today = datetime.today()
    for i in range(len(weather.index)):
        dto = datetime.strptime(weather.loc[i,'date'], formatter_string).replace(year = today.year)      
        if (date(2017,12,21) <= dto.date() <= date(2017,12,31) or date(2017,1,1) <= dto.date() <= date(2017,3,20)):
            weather.loc[i,'season'] = "Winter"
        if (date(2017, 3, 21) <= dto.date() <= date(2017, 6, 21)):
            weather.loc[i,'season'] = "Spring"
        if (date(2017, 6, 22) <= dto.date() <= date(2017, 9, 21)):
            weather.loc[i,'season'] = "Summer"
        if (date(2017, 9, 22) <= dto.date() <= date(2017, 12, 20)):
            weather.loc[i,'season'] = "Fall"
            
        names = ["Cold", "Cool", "Mild", "Warm", "Hot"]
        names_idx = [1, 2, 3, 4, 5]
        Temperature_bins = [-50, 10, 16, 30, 35, 1000]
        [1,3,5]
        if pd.isnull(weather.loc[i,'TAVG']):
            weather.loc[i,'TAVG'] = (weather.loc[i,'TMIN'] + weather.loc[i,'TMAX'])/2
        weather['TCAT'] = pd.cut(weather['TAVG'], Temperature_bins, labels = names)
        weather['TCAT_idx'] = pd.cut(weather['TAVG'], Temperature_bins, labels = names_idx)
              
    return(weather)    
    
    
def do_sentiment_analysis(reviews):
    for i in range(len(reviews.index)):       
        analysis = TextBlob(str(reviews.loc[i,'text']))
        ## Check polarity to determine sentiment
        if analysis.sentiment.polarity > 0:
            reviews.loc[i,'sentiment'] = 'positive'
            reviews.loc[i,'sentiment_idx'] = 1
            reviews.loc[i,'sentiment_polarity'] = round(analysis.sentiment.polarity,2)
        elif analysis.sentiment.polarity == 0:
            reviews.loc[i,'sentiment'] = 'neutral'
            reviews.loc[i,'sentiment_idx'] = 0
            reviews.loc[i,'sentiment_polarity'] = round(analysis.sentiment.polarity,2)
        else:
            reviews.loc[i,'sentiment']='negative'
            reviews.loc[i,'sentiment_idx'] = -1
            reviews.loc[i,'sentiment_polarity'] = round(analysis.sentiment.polarity,2)
    return reviews


def draw_word_cloud(reviews, sentiment):
    if sentiment != "all":
        reviews = reviews[reviews['sentiment'] == sentiment]  
    reviews['text'].to_csv(r'reviewsText.txt', header = True, index = None, sep='|', mode='a')
    with open('reviewsText.txt','r') as file:
        rev = file.read()
        file.close
    clean_tokens = []
    stopWords = set(stopwords.words('english'))
    ## Use nltk.tokenize.word_tokenize to collect words
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(rev)
    for word in words:
        word = word.lower()
        if word not in stopWords:
            if word.isalpha(): # Drop all non-words
                clean_tokens.append(word)
    ## Use fdist to calculate the frequency of words
    fdist = FreqDist(clean_tokens)
    ## Sort in descending order of frequency
    vocabulary = fdist.most_common(fdist.N())
    vocab = pd.DataFrame(vocabulary, columns = ["words", "frequency"])  
    wordcloud = WordCloud(background_color = "white", width = 800, height = 300).generate(' '.join(vocab['words']))
    fig = plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    fig.savefig(sentiment + '_WordCloud.png')
    
    
###-- basic statistical analysis (10 attributes)

#Determine the mean (mode if categorical), median, 
#and standard deviation of at least 10 attributes in your data sets. 
#Use Python to generate these results and use the project report to show and explain each.
def my_summary(x):
    print("Count:   ", len(x))
    print("Missing: ", sum((x).isnull()))
    print("Mean:    ", round(np.mean(x),1))
    print("Median:  ", round(np.median(x),1))
    print("Max:     ", round(np.amax(x),1))
    print("Min:     ", round(np.amin(x),1))
    print("Std:     ", round(np.std(x),1))
    print("Type:    ", x.dtype.name)
    print("\n")
    
def my_summary_cat(x):
    print("Count:   ", len(x))
    print("Missing: ", sum((x).isnull()))
    print("Mode:    ", mode(x))
    print("Median:  ", np.median(x))
    print("Max:     ", np.amax(x))
    print("Min:     ", np.amin(x))
    print("Std:     ", round(np.std(x),1))
    print("Type:    ", x.dtype.name)  
    print("\n")

def my_summary_cat_2(x):
    print("Count:   ", len(x))
    print("Missing: ", sum((x).isnull()))
    print("Mode:    ", mode(x))
    print("Uniques: ", len((x).unique()))
    print("Type:    ", x.dtype.name)


def run_summary_stats(df):
    #1# temperature (in celcius)
    my_summary(df['TAVG'])

    #2# percipitation
    my_summary(df['PRCP'])

    #3# snow
    # Median calculation might show warnings due to NaNs in the data
    my_summary(df['SNOW'])

    #4# review ratings
    my_summary_cat(df['review_rating'])
    df['review_rating'].value_counts()

    #5# restaurant ratings
    my_summary(df['rest_rating'])

    #6# restaurant review count
    my_summary(df['review_count'])

    #7# sentiment (1=positive, -1=negative, 0=neutral)
    my_summary_cat(df['sentiment_idx'])

    #8# sentiment polarity
    my_summary(df['sentiment_polarity'])

    #9# distance (in miles)
    my_summary(df['dist'])

    #10#
    my_summary_cat_2(df['state'])

    #11#
    my_summary_cat_2(df['price'])

    
    
def draw_histograms(df):
    ###-- histograms

    #Use a histogram to plot at least three (3) of the variables (attributes) in either dataset. 
    #Discuss the insight generated by the histograms. What do they show or suggest?

    #1# PRCP
    sns.distplot(df['PRCP'])
    plt.title('Histogram: PRCP')
    plt.xlabel('PRCP (in tenths of cm)')
    plt.ylabel('Frequency')

    #2# TAVG
    sns.distplot(df['TAVG'])
    plt.title('Histogram: TAVG')
    plt.xlabel('Temp (in C)')
    plt.ylabel('Frequency')

    #3# SNOW 
    sns.distplot(df['SNOW'][df['SNOW'] > 0])
    plt.title('Histogram: SNOW')
    plt.xlabel('SNOW (in tenths of mm)')
    plt.ylabel('Frequency')
    

def draw_scatterplots(df):
    ###-- scatterplots and correlations

    #Identify three (3) quantitative variables from either data set. 
    #Find the correlation between all the pairs of these quantity variables. 
    #Include a table of the output in your report, and explain your findings 
    #– what does this indicate about your data? 
    #Use scatterplots to display the results. Ideally, create a set of scatterplot subplots

    #3 variables: review_rating(ordinal), TAVG(continuous), sentiment_polarity(continuous)

    #correlation matrix - quite low
    df_subset = df[['TAVG','sentiment_polarity','review_rating']]
    df_subset.corr()

    sns.pairplot(df_subset)

    #1# scatter: sentiment_polarity vs. review_rating (with noise)
    sns.stripplot(x=df['review_rating'], y=df['sentiment_polarity'], jitter=True, alpha=0.25)
    plt.title('Sentiment Polarity vs. Yelp Rating')
    plt.xlabel('Yelp Rating')
    plt.ylabel('Sentiment Polarity')
    plt.show()

    #crosstab
    pd.crosstab(index=df['sentiment_idx'], 
            columns=df['review_rating']).apply(lambda r: round(r/r.sum(),2), axis=0)


    #2# scatters: sentiment_polarity vs. TAVG by region
    g = sns.FacetGrid(df, col='region', col_wrap=3)
    g.map(plt.scatter, 'sentiment_polarity', 'TAVG', alpha=0.25)

 
    #3# scatters: review_rating vs. TAVG by region
    g = sns.FacetGrid(df, col='region', col_wrap=3)
    g.map(sns.stripplot, 'review_rating', 'TAVG', alpha=0.25, jitter=True)


def do_kmeans(all_data):
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
    print("For k (number of clusters) = ", k, ", the average silhouette_score is :", round(silhouette_avg, 3))
    ## Convert our high dimensional data to 2 dimensions using PCA
    pca2D = decomposition.PCA(2)
    ## Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    fig = plt.figure(figsize=(15,8))
    ## Plot using a scatter plot and shade by cluster label
    #plt.scatter(x = plot_columns[:,0], y = plot_columns[:,1], c = cluster_labels)
    unique_labels = set( cluster_labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]

    for m, col in zip(unique_labels, colors):
            class_member_mask = (cluster_labels == m)
            xy = plot_columns[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                        markeredgecolor='m', markersize=14)
    plt.title('KMeans with '+ str(k) + ' clusters (silhouette score = ' + str(round(silhouette_avg,3)) + ')')   
    fig.savefig('KMeans_Clusters_' + str(k) + '.jpg')
    plt.show()
    
    
def do_dbscan(df):
    dt = pd.concat([df['rest_lat'], df['rest_lon'], df['TAVG'], df['sentiment_polarity']], 
                   axis = 1, keys = ['rest_lat', 'rest_lon', 'TAVG', 'sentiment_polarity'])
    x = dt.values
    #Normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)

    i=0.4
    j=30
    db = DBSCAN(eps=i, min_samples=j).fit(normalizedDataFrame) 
    db_labels = db.labels_

    #--calculate the labels
    #If there are outliers, then the number of data is the length of label set -1
    #--otherwise the length of label set -1
    if(-1 in db_labels):
        db_num_clusters=len(set(db_labels))-1
    else:
        db_num_clusters=len(set(db_labels))

    #Calculate the sihouette score
    db_sihouette_score = metrics.silhouette_score(x, db_labels)
    print("Silhouette Score=", round(db_sihouette_score,3))
    print("Number of clusters=", db_num_clusters)
    print("eps=",i)
    print("min_samples=",j)
    pca2D = decomposition.PCA(2)
    # Turn the data into two columns with PCA
    fig = plt.figure(figsize=(15,8))
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    unique_labels = set(db_labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
            class_member_mask = (db_labels == k)
            xy = plot_columns[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                        markeredgecolor='k', markersize=14)
    plt.title('DBSCAN with eps '+ str(i) + ' and min_samples '+ str(j)+' (sihouette score = ' + str(round(db_sihouette_score, 3)) + ')')   
    fig.savefig('DBSCAN.jpg')
    plt.show()


def hierarchCluster(df):
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
    S = cosine_similarity(bag)
    lnk = ward(S)
    
    #evaluate the quality of ward cluster using silhouette scores
    ward_label = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
    w_label = ward_label.fit_predict(lnk)
    
    h_silhouette_score = silhouette_score(lnk, w_label)
    
    fig = plt.figure(figsize=(15,8))
    plt.title('Hierarchical Clustering Dendrogram of Reviews by Region (sihouette score = ' + str(round(h_silhouette_score, 2)) + ')')
    plt.xlabel('cosine similarity distance')
    plt.ylabel('Region')
    dendrogram(lnk, orientation = 'right', labels = region)
    plt.tight_layout()
    fig.savefig('HierarchicalClustering.jpg')
    plt.show()
    

def do_apriori():
    ## import a list of all major food items that may be available in restaurants
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

    df = pd.read_csv('merged_data.csv', encoding = 'Latin1')
    ## keep only 5-star ratings
    df = df[df['review_rating'] == 5]
    df = df.reset_index(drop = True)

    ## tokenize the review text and store it in a new column
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
    ## Pick up the 5 highest frequency words
    #voca = pd.DataFrame(fdist.most_common(5))
    voca = pd.DataFrame(fdist.most_common(30))
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
    def get_support(df, sample_space):
        pp = []
        for cnum in range(1, len(df.columns)+1):
            for cols in combinations(df, cnum):
                s = df[list(cols)].all(axis=1).sum()
                #pp.append([",".join(cols), s, round(s/sample_space, 5)])
                pp.append([",".join(cols), s, round(s, 5)])
        sdf = pd.DataFrame(pp, columns=['Pattern', 'Count', 'Support'])
        return sdf


    sample_space = len(df)
    s = get_support(voca_pivot, sample_space)

    ## find the itemsets that occur frequently
    ss = s[s['Pattern'].str.contains(',')]
    freq_itemsets = ss.sort_values('Support', ascending = False)
    freq_itemsets = freq_itemsets[freq_itemsets['Count'] > 0]
    freq_itemsets = freq_itemsets.reset_index(drop = True)


    ## calculate the confidence for the top 8 most frequently occuring itemsets
    asso_rules = pd.DataFrame(columns=['Association', 'Support', 'Confidence'], 
                          index = freq_itemsets.head(8).index.copy())
    for i in range(len(freq_itemsets.head(8).index)):
        ii = freq_itemsets.loc[i,'Pattern'].split(",")
        asso_rules.loc[i,'Association'] = ii[0] + '-->' + ii[1]
        sup = freq_itemsets.loc[i,'Support']
        asso_rules.loc[i,'Support'] = sup
        condition_count = int(s[s['Pattern'] == ii[0]]['Count'])
        conf = float(round(freq_itemsets.loc[i,'Count']/condition_count, 5))
        asso_rules.loc[i,'Confidence'] = conf

    ## print the support and confidence levels for all the frequent itemsets
    pprint(asso_rules)

    ## print the support and confidence levels for all the frequent itemsets
    ## with support level greater than 0.25%
    print()
    print("---0.25% support---")
    pprint(asso_rules[asso_rules["Support"] >= 0.0025])

    ## print the support and confidence levels for all the frequent itemsets
    ## with support level greater than 0.15%
    print()
    print("---0.15% support---")
    pprint(asso_rules[asso_rules["Support"] >= 0.0015])

    ## print the support and confidence levels for all the frequent itemsets
    ## with support level greater than 0.10%
    print()
    print("---0.10% support---")
    pprint(asso_rules[asso_rules["Support"] >= 0.0010])
    

if __name__ == "__main__":
    ## Execute only if run as a script  
    main(sys.argv)