import numpy as np # linear algebra
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import svm

news = pd.read_csv("fake.csv")

articles1 = pd.read_csv("articles1.csv")
articles2 = pd.read_csv("articles2.csv")
articles3 = pd.read_csv("articles3.csv")

articles = pd.concat([articles1, articles2, articles3])
articles = articles.sample(n=13000)
articles.rename(columns={'url': 'site_url', 'content': 'text'}, inplace=True)
articles["type_id"] = 0
articles["language"] = "english"

# news["country_id"] = 0
# news["country_id"][news["country"]=="US"] = 1
# news["country_id"][news["country"]=="CO"] = 2
# news["country_id"][news["country"]=="FR"] = 3
# news["country_id"][news["country"]=="DE"] = 4
# news["country_id"][news["country"]=="GB"] = 5
# news["country_id"][news["country"]=="CA"] = 6
# news["country_id"][news["country"]=="AU"] = 7
# news["country_id"][news["country"]=="EU"] = 8
# news["country_id"][news["country"]=="NL"] = 9
# news["country_id"][news["country"]=="LI"] = 10
# news["country_id"][news["country"]=="SG"] = 11
# news["country_id"][news["country"]=="IO"] = 12
# news["country_id"][news["country"]=="ME"] = 13
# news["country_id"][news["country"]=="TV"] = 14
# news["country_id"][news["country"]=="ES"] = 15
# news["country_id"][news["country"]=="RU"] = 16
# news["country_id"][news["country"]=="IN"] = 17
# news["country_id"][news["country"]=="US"] = 18
# news["country_id"][news["country"]=="US"] = 19
# news["country_id"][news["country"]=="EE"] = 20
# news["country_id"][news["country"]=="SE"] = 21
# news["country_id"][news["country"]=="ZA"] = 22
# news["country_id"][news["country"]=="IS"] = 23
# news["country_id"][news["country"]=="BG"] = 24
# news["country_id"][news["country"]=="CH"] = 25

news["type_id"] = 1
# news["type_id"][news["type"]=="bias"] = 1
# news["type_id"][news["type"]=="conspiracy"] = 2
# news["type_id"][news["type"]=="fake"] = 3
# news["type_id"][news["type"]=="bs"] = 4
# news["type_id"][news["type"]=="satire"] = 5
# news["type_id"][news["type"]=="hate"] = 6
# news["type_id"][news["type"]=="junksci"] = 7
# news["type_id"][news["type"]=="state"] = 8

news_final = pd.concat([articles, news], ignore_index=True)

news_final["count"] = news_final["text"].str.len()

print(news_final.describe())
print(news_final[news_final["type_id"]==0].describe())
print(news_final[news_final["type_id"]==1].describe())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(news_final["text"][news_final["language"]=="english"], news_final["type_id"][news_final["language"]=="english"], test_size=0.2, random_state=53)

count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
count_train = count_vectorizer.fit_transform(X_train.values.astype('U'))

# Transform the test set
count_test = count_vectorizer.transform(X_test.values.astype('U'))

# Initialize the `tfidf_vectorizer`
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train.values.astype('U'))

# Transform the test set
tfidf_test = tfidf_vectorizer.transform(X_test.values.astype('U'))

clf = MultinomialNB()
clf_svm = svm.SVC(kernel='linear', C = 1.0)

clf.fit(count_train, y_train)
clf_svm.fit(count_train, y_train)
pred = clf.predict(count_test)
pred_svm = clf_svm.predict(count_test)

score = metrics.accuracy_score(y_test, pred)
print("Accuracy for Count Vectorizer:   %0.3f" % score)

score = metrics.accuracy_score(y_test, pred_svm)
print("Accuracy for Count Vectorizer (SVM):   %0.3f" % score)

clf.fit(tfidf_train, y_train)
clf_svm.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
pred_svm = clf_svm.predict(count_test)

score = metrics.accuracy_score(y_test, pred)
print("Accuracy for TFIDF Vectorizer:   %0.3f" % score)

score = metrics.accuracy_score(y_test, pred_svm)
print("Accuracy for TFIDF Vectorizer (SVM):   %0.3f" % score)
