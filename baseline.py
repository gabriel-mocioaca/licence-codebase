from pre_process import preprocess
import pandas as pd
import numpy
import timeit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

train_dataset = pd.read_csv('corona_nlp_train.csv', header=0, delimiter=',', usecols=['OriginalTweet', 'Sentiment'])
test_dataset = pd.read_csv('corona_nlp_test.csv', header=0, delimiter=',', usecols=['OriginalTweet', 'Sentiment'])

train_features = preprocess(train_dataset)
test_features = preprocess(test_dataset)

train_labels = train_dataset.pop('Sentiment')
test_labels = test_dataset.pop('Sentiment')

train_labels= train_labels.replace({'Extremely Positive':'Positive','Extremely Negative':'Negative'})
test_labels = test_labels.replace({'Extremely Positive':'Positive','Extremely Negative':'Negative'})

print("count_nb")
count_vectorizer = CountVectorizer()
clf_nb = MultinomialNB()

X = count_vectorizer.fit_transform(train_features)
Y = numpy.asarray(train_labels, dtype="|S6")
x = count_vectorizer.transform(test_features)
y = numpy.asarray(test_labels, dtype="|S6")

time_train = timeit.timeit(lambda: clf_nb.fit(X, Y), number=1)
print(f"time_train: {'%.4f'%time_train}")
print(clf_nb.score(x, y))

print("tfid_nb")
tfid_vectorizer = TfidfVectorizer()
clf_nb = MultinomialNB()

X = tfid_vectorizer.fit_transform(train_features)
Y = numpy.asarray(train_labels, dtype="|S6")
x = tfid_vectorizer.transform(test_features)
y = numpy.asarray(test_labels, dtype="|S6")

time_train = timeit.timeit(lambda: clf_nb.fit(X, Y), number=1)
print(f"time_train: {'%.4f'%time_train}")
print(clf_nb.score(x, y))

print("count_rfc")
count_vectorizer = CountVectorizer()
clf_rfc = RandomForestClassifier()

X = count_vectorizer.fit_transform(train_features)
Y = numpy.asarray(train_labels, dtype="|S6")
x = count_vectorizer.transform(test_features)
y = numpy.asarray(test_labels, dtype="|S6")

time_train = timeit.timeit(lambda: clf_rfc.fit(X, Y), number=1)
print(f"time_train: {'%.4f'%time_train}")
print(clf_rfc.score(x, y))

print("tfid_rfc")
count_vectorizer = TfidfVectorizer()
clf_rfc = RandomForestClassifier()

X = count_vectorizer.fit_transform(train_features)
Y = numpy.asarray(train_labels, dtype="|S6")
x = count_vectorizer.transform(test_features)
y = numpy.asarray(test_labels, dtype="|S6")

time_train = timeit.timeit(lambda: clf_rfc.fit(X, Y), number=1)
print(f"time_train: {'%.4f'%time_train}")
print(clf_rfc.score(x, y))

print("count_svc")
count_vectorizer = CountVectorizer()
clf_svc = SVC()

X = count_vectorizer.fit_transform(train_features)
Y = numpy.asarray(train_labels, dtype="|S6")
x = count_vectorizer.transform(test_features)
y = numpy.asarray(test_labels, dtype="|S6")

time_train = timeit.timeit(lambda: clf_svc.fit(X, Y), number=1)
print(f"time_train: {'%.4f'%time_train}")
print(clf_svc.score(x, y))

print("tfid_svc")
count_vectorizer = TfidfVectorizer()
clf_svc = SVC()

X = count_vectorizer.fit_transform(train_features)
Y = numpy.asarray(train_labels, dtype="|S6")
x = count_vectorizer.transform(test_features)
y = numpy.asarray(test_labels, dtype="|S6")

time_train = timeit.timeit(lambda: clf_svc.fit(X, Y), number=1)
print(f"time_train: {'%.4f'%time_train}")
print(clf_svc.score(x, y))