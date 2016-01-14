import numpy as np
import pandas as pd
import seaborn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
import re
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora, models
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import date, datetime
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer as DV
import operator
from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.externals import joblib

stoplist = set(stopwords.words("english"))

#Sentence sentiment
class TextSentiment(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self
  def transform(self, reviews):
  #create an array with polarity, intensity, subjectivity
    textblob_output = []
    for review in reviews:
      tb = TextBlob(review)
      temp = []
      temp.append(tb.sentiment.polarity)
      temp.append(tb.sentiment.subjectivity)
      textblob_output.append(temp)
    return textblob_output

class LDATopics(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None, topics = 15):
    lda = perform_lda(x, topics)
    self.lda = lda
    return self
  def transform(self, reviews, topics=15):
  #create an array with topic distribution
    lda = self.lda
    topic_dist_list = []
    topic_dist_list = generate_topic_dist_matrix(lda, topics, reviews, topic_dist_list)
    cols = []
    for i in xrange(1, topics+1):
      cols.append("Topic"+ str(i))
    topic_dist_df = pd.DataFrame(topic_dist_list, columns=cols)
    features = list(topic_dist_df.columns[:topics])
    x_train = topic_dist_df[features]
    #print x_train.head(3)
    return map(list,x_train.values)
  def returnLDA(self):
    return self.lda

def perform_lda(allReviewsTrain, numTopics):
    corpus = []
    for review in allReviewsTrain:
        # Remove punctuations
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
        texts = [word for word in review.lower().split() if word not in stoplist]
        try:
            corpus.append(texts)
        except:
            pass

    # Build dictionary
    dictionary = corpora.Dictionary(corpus)
    dictionary.save('restaurant_reviews.dict')
        
    # Build vectorized corpus
    corpus_2 = [dictionary.doc2bow(text) for text in corpus]
    
    lda = models.LdaModel(corpus_2, num_topics=numTopics, id2word=dictionary)
    return lda

def generate_topic_dist_matrix(lda, numTopics, corpus, all_dist):
    topic_dist = [0] * numTopics
    dictionary = corpora.Dictionary.load("restaurant_reviews.dict")
    for doc in corpus:
        vec = dictionary.doc2bow(doc.lower().split())
        output = lda[vec]
        highest_prob = 0
        highest_topic = 0
        temp = [0] * numTopics    # List to keep track of topic distribution for each document
        for topic in output:
            this_topic, this_prob = topic
            temp[this_topic] = this_prob
            if this_prob > highest_prob:
                highest_prob = this_prob 
                highest_topic = this_topic
        
        all_dist.append(temp)
        topic_dist[highest_topic] += 1
    return all_dist

def process_reviews(dirty_data_set):
    clean_data_set = []
    for review in dirty_data_set:
        # Remove punctuations
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
        texts = [word for word in review.lower().split() if word not in stoplist]
        try:
            clean_data_set.append(' '.join(texts))
        except:
            pass
    return clean_data_set


#spring = march 20 to june 20
#summer = june 20 to sep 20
#autum = sep 20 to dec 20
#winter = dec 20 to mar 20

def friends_len(flist):
  return len(list(flist))

def elite_len(elist):
  return len(list(elist))

#### Function to conert df into integers - START $$$$
def convert_to_integer(srs):
  d = get_nominal_integer_dict(srs)
  return srs.map(lambda x: d[x])

def get_nominal_integer_dict(nominal_vals):
  d = {}
  for val in nominal_vals:
    if val not in d:
      current_max = max(d.values()) if len(d) > 0 else -1
      d[val] = current_max+1
  return d

def convert_strings_to_integer(df):
  ret = pd.DataFrame()
  for column_name in df:
    column = df[column_name]
    if column.dtype=='string' or column.dtype=='object':
      ret[column_name] = convert_to_integer(column)
    else:
      ret[column_name] = column
  return ret
#### Function to conert df into integers - END $$$$

def processFacilities(minrl,maxrl,topicsSize,topicLen,season):
  #read datasets
  business_data = pd.read_csv('../yelp_academic_dataset_business.csv', dtype=unicode)
  #checkin_data = pd.read_csv('../yelp_academic_dataset_checkin.csv')
  review_data = pd.read_csv('../yelp_academic_dataset_review.csv')
  #tip_data = pd.read_csv('../yelp_academic_dataset_tip.csv')
  user_data = pd.read_csv('../yelp_academic_dataset_user.csv')

  #Merge dataframes using pandas
  review_user_data = review_data.merge(user_data,
                                       left_on='user_id',
                                       right_on='user_id',
                                       how='outer',
                                       suffixes=('_review', '_user'))

  business_review_user_data = review_user_data.merge(business_data,
                                                     left_on='business_id',
                                                     right_on='business_id',
                                                     how='outer',
                                                     suffixes=('_reviewuser', '_business'))


  #rename the columns
  business_review_user_data = business_review_user_data.rename(columns = {'name_reviewuser':'name_user',
                                                                          'review_count_reviewuser':'review_count_user',
                                                                          'stars_reviewuser':'stars_review'})

  cols = business_review_user_data.columns
  cols = cols.map(lambda x: x.replace(' ', '_').lower() if isinstance(x, (str, unicode)) else x)
  business_review_user_data.columns = cols


  rDF = business_review_user_data[business_review_user_data['categories'].str.contains('Restaurants')]

  #Drop duplicate columns
  rDF = rDF.drop('attributes.good_for_kids',axis=1)

  #Stop words from corpus
  stoplist = set(stopwords.words("english"))
  numTopics = 15

  
  #Stop words from corpus
  stoplist = set(stopwords.words("english"))
  # #Drop empty attributes
  t = rDF.dropna(how='all',axis=1)
  t = rDF
  minReviewLen = int(minrl)
  maxReviewLen = int(maxrl)
  reviewSelected = len(t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen])

  print "Number of reviews selected before applying Season filter : ",reviewSelected
  rDF = t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]

  #Convert Date to Month and drop date
  rDF['date_month'] = pd.DatetimeIndex(rDF['date']).month

  #For topics and rating prediction
  firstFewRows = rDF.ix[:,['categories','text','stars_review','business_id','date_month']]
  rDF2 = firstFewRows[firstFewRows['categories'].str.contains('Restaurants')]

  
  ## To remove given columns from data frame
  rDF = rDF.drop(['text','date','review_id','neighborhoods','categories','full_address'],axis=1)

  month_list = [1,2,3,4]
  if season == 0:
    month_list = [1,2,3,4,5,6,7,8,9,10,11,12]
  elif season == 1:
    month_list = [1,2,3]
  elif season == 2:
    month_list = [4,5,6]
  elif season == 3:
    month_list = [7,8,9]
  else:
    month_list = [10,11,12]

  rDF = rDF[rDF['date_month'].isin(month_list)]
  rDF2 = rDF2[rDF2['date_month'].isin(month_list)]

  reviewSelected = len(rDF.index)
  print 'No. of reviews selected after Season filter : ', reviewSelected

  print 'converting friends and elite'

  #Convert friends and elite to numbers
  rDF['friends_len'] = rDF['friends'].apply(lambda x: friends_len(x))
  rDF['elite_len'] = rDF['elite'].apply(lambda x: elite_len(x))

  rDF = rDF.drop(['friends','elite'],axis=1)
  print 'After dropped f and E ############'


  # Drop reviews with NaN value in stars_review
  print 'No of reviews (Before dropping NaaN): ', len(rDF.index),len(rDF2.index)
  rDF = rDF[np.isfinite(rDF['stars_review'])]
  rDF2 = rDF2[np.isfinite(rDF2['stars_review'])]
  print 'No of reviews (After dropping NaaN): ',len(rDF.index),len(rDF2.index)

  rDF = rDF.apply(pd.to_numeric, errors='ignore')

  #Drop empty attributes
  rDF = rDF.dropna(how='all',axis=1)


  #Convert attributes value into numeric
  df = convert_strings_to_integer(rDF)

  df_y = df['stars_review'].tolist()

  df = df.drop('stars_review',axis=1)


  cols_list = list(df.columns.values)


  df_x = df.values.tolist()

  # x is list of lists from pandas
  im = Imputer(missing_values='NaN',
              strategy="mean",
              axis=0)
  x = im.fit_transform(df_x)
  print x.shape
  # y = im.fit_transform(y_train)
  anova_k = SelectKBest(f_classif, k=5)
  anova_k.fit_transform(x, df_y)

  print '**** Anova Scores ******'
  print anova_k.scores_

  list_scores = anova_k.scores_

  cols_scores = {}
  for i in range(len(cols_list)):
    if str(list_scores[i]) != 'nan':
      cols_scores[cols_list[i]] = list_scores[i]
    else:
      continue


  #Sort the score dictionary
  print '*** After sorted ****'
  sorted_cols = OrderedDict(sorted(cols_scores.items(), key=lambda t: t[1],reverse=True))
  print sorted_cols

  sorted_cols_values = list(sorted_cols.values())
  keys_list = list(sorted_cols.keys())
  print '**** Scaled ****'

  total = sum(sorted_cols_values)

  for i, val in enumerate(sorted_cols_values):
    sorted_cols[keys_list[i]] = round((val/total)*100,2)

  print sorted_cols
  print "\n"

  pipeline = Pipeline([
      ("Imputer", Imputer(missing_values='NaN',
                strategy="mean",
                axis=1)),
      ('features', FeatureUnion([
      ('best', Pipeline([
      # ('one_hot_encoding', OneHotEncoder(categorical_features=)),
      ('anova', SelectKBest(f_classif, k=25)),
      ('pca', decomposition.PCA(n_components=25))

      ]))])),
      # ('classifier', SVC())
      ('classifier', GaussianNB())
    ])

  df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.10)

  pipeline.fit(df_x_train, df_y_train)

  preds = pipeline.predict(df_x_test)

  Results = {}
  precision = metrics.precision_score(df_y_test, preds)
  recall = metrics.recall_score(df_y_test, preds)
  f1 = metrics.f1_score(df_y_test, preds)
  accuracy = accuracy_score(df_y_test, preds)

  data = {'precision':precision,
              'recall':recall,
              'f1_score':f1,
              'accuracy':accuracy}

  Results['clf'] = data
  cols = ['precision', 'recall', 'f1_score', 'accuracy']
  print pd.DataFrame(Results).T[cols].T

  print 'fit finished'

  # # return the facilites
  # return sorted_cols, reviewSelected
# Convert dataframe values into lists


  ## Grouping the review based on star rating
  starsGroup = rDF2.groupby('stars_review')

  all_1stars_text = starsGroup.get_group(1.0)['text']
  all_2stars_text = starsGroup.get_group(2.0)['text']
  all_3stars_text = starsGroup.get_group(3.0)['text']
  all_4stars_text = starsGroup.get_group(4.0)['text']
  all_5stars_text = starsGroup.get_group(5.0)['text']

  all_1stars_labels = [1.0]*len(all_1stars_text)
  all_2stars_labels = [2.0]*len(all_2stars_text)
  all_3stars_labels = [3.0]*len(all_3stars_text)
  all_4stars_labels = [4.0]*len(all_4stars_text)
  all_5stars_labels = [5.0]*len(all_5stars_text)

  ## Split test and train reviews
  all_1stars_text_train, all_1stars_text_test, all_1stars_labels_train, all_1stars_labels_test = train_test_split(all_1stars_text, all_1stars_labels, test_size=0.10)
  all_2stars_text_train, all_2stars_text_test, all_2stars_labels_train, all_2stars_labels_test = train_test_split(all_2stars_text, all_2stars_labels, test_size=0.10)
  all_3stars_text_train, all_3stars_text_test, all_3stars_labels_train, all_3stars_labels_test = train_test_split(all_3stars_text, all_3stars_labels, test_size=0.10)
  all_4stars_text_train, all_4stars_text_test, all_4stars_labels_train, all_4stars_labels_test = train_test_split(all_4stars_text, all_4stars_labels, test_size=0.10)
  all_5stars_text_train, all_5stars_text_test, all_5stars_labels_train, all_5stars_labels_test = train_test_split(all_5stars_text, all_5stars_labels, test_size=0.10)

  ##Pre processing the review text
  # Process the reviews
  corpus_5stars = process_reviews(all_5stars_text_train)
  corpus_4stars = process_reviews(all_4stars_text_train)
  corpus_3stars = process_reviews(all_3stars_text_train)
  corpus_2stars = process_reviews(all_2stars_text_train)
  corpus_1stars = process_reviews(all_1stars_text_train)

  # print "Number of 5-star reviews after processing: ", len(corpus_5stars)
  # print "Number of 4-star reviews after processing: ", len(corpus_4stars)
  # print "Number of 3-star reviews after processing: ", len(corpus_3stars)
  # print "Number of 2-star reviews after processing: ", len(corpus_2stars)
  # print "Number of 1-star reviews after processing: ", len(corpus_1stars)

  all_5_4_train = np.append(corpus_5stars, corpus_4stars)
  all_5_4_3_train = np.append(all_5_4_train, corpus_3stars)
  all_5_4_3_2_train = np.append(all_5_4_3_train, corpus_2stars)
  all_text_train = np.append(all_5_4_3_2_train, corpus_1stars)
  pickle.dump(all_text_train, open("all_text_train.p", "wb"))

  ## Create training labels
  all_5_4_label = np.append(all_5stars_labels_train,all_4stars_labels_train)
  all_5_4_3_label = np.append(all_5_4_label,all_3stars_labels_train)
  all_5_4_3_2_label = np.append(all_5_4_3_label,all_2stars_labels_train)
  all_label_train = np.append(all_5_4_3_2_label,all_1stars_labels_train)
  pickle.dump(all_label_train, open("all_label_train.p", "wb"))

  # Process the test reviews
  corpus_5stars = process_reviews(all_5stars_text_test)
  corpus_4stars = process_reviews(all_4stars_text_test)
  corpus_3stars = process_reviews(all_3stars_text_test)
  corpus_2stars = process_reviews(all_2stars_text_test)
  corpus_1stars = process_reviews(all_1stars_text_test)

  # print "Number of 5-star reviews after processing: ", len(corpus_5stars)
  # print "Number of 4-star reviews after processing: ", len(corpus_4stars)
  # print "Number of 3-star reviews after processing: ", len(corpus_3stars)
  # print "Number of 2-star reviews after processing: ", len(corpus_2stars)
  # print "Number of 1-star reviews after processing: ", len(corpus_1stars)

  all_5_4_test = np.append(corpus_5stars, corpus_4stars)
  all_5_4_3_test = np.append(all_5_4_test, corpus_3stars)
  all_5_4_3_2_test = np.append(all_5_4_3_test, corpus_2stars)
  all_text_test = np.append(all_5_4_3_2_test, corpus_1stars)
  pickle.dump(all_text_test, open("all_text_test.p", "wb"))


  ## Create test labels
  all_5_4_label = np.append(all_5stars_labels_test,all_4stars_labels_test)
  all_5_4_3_label = np.append(all_5_4_label,all_3stars_labels_test)
  all_5_4_3_2_label = np.append(all_5_4_3_label,all_2stars_labels_test)
  all_label_test = np.append(all_5_4_3_2_label,all_1stars_labels_test)
  pickle.dump(all_label_test, open("all_label_test.p", "wb"))


  #Generate tf-idf vectors
  vectorizer = TfidfVectorizer()
  senti = TextSentiment()
  lda = LDATopics()

  lda.fit(all_text_train)
  topic_word = lda.returnLDA()
  #print topic_word.print_topics(num_topics=50, num_words=1)
  topicsList = topic_word.show_topics(num_topics=topicsSize, num_words=topicLen, log=False, formatted=True)
  print topicsList
  # print topic_word.top_topics(corpus, num_words=20)
  


  combined_features = FeatureUnion([("tfidf", vectorizer), ("sentiment", senti),("LDA", lda)])

  #rf = svm.SVC(class_weight="balanced",kernel="linear",gamma="auto")
  rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

  pipeline = Pipeline([("features", combined_features),( "rf", rf)])
  pipeline.fit(all_text_train, all_label_train)
  preds = pipeline.predict(all_text_test)

  #Dump the model
  pickle.dump(pipeline,open('SVCmodel.p','wb'))

  Results = {}
  precision = metrics.precision_score(all_label_test, preds)
  recall = metrics.recall_score(all_label_test, preds)
  f1 = metrics.f1_score(all_label_test, preds)
  accuracy = accuracy_score(all_label_test, preds)

  data = {'precision':precision,
              'recall':recall,
              'f1_score':f1,
              'accuracy':accuracy}

  Results['clf'] = data
  cols = ['precision', 'recall', 'f1_score', 'accuracy']
  print pd.DataFrame(Results).T[cols].T
  print "********* Model Saved ********** "

  return sorted_cols, reviewSelected,topicsList

def getReview(rText):
  clf = pickle.load(open('SVCmodel.p','rb'))
  rating = clf.predict([rText])
  print 'Rating ********** >', rating
  return rating

# return sorted_cols, reviewSelected

