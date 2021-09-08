import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


train =  pd.read_csv('corona_nlp_train.csv')


sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11,4)})
sns.countplot(train['Sentiment'])
plt.show()

print(train.Sentiment.value_counts())

train.Sentiment=train.Sentiment.replace({'Extremely Positive':'Positive','Extremely Negative':'Negative'})

sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11,4)})
sns.countplot(train['Sentiment'])
plt.show()

print(train.Sentiment.value_counts())