import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pre_process import preprocess

train_data = pd.read_csv('corona_nlp_train.csv')
print("Data shape -> ", train_data.shape)

train_data = train_data[['OriginalTweet', 'Sentiment']]
print("Data shape -> ", train_data.shape)
print(train_data.head(5))

print(train_data.isnull().sum())

for index,text in enumerate(train_data['OriginalTweet'][5:10]):
  print('Tweet %d:\n'%(index+1),text)

train_data['lem'] = preprocess(train_data)

for index,text in enumerate(train_data['lem'][5:10]):
  print('Tweet %d:\n'%(index+1),text)



df_grouped=train_data[['Sentiment','lem']].groupby(by='Sentiment').agg(lambda x:' '.join(x))
print(df_grouped.head())

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df_grouped['lem'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index=df_grouped.index
df_dtm.head(3)

from wordcloud import WordCloud
from textwrap import wrap

def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()
  
df_dtm=df_dtm.transpose()

for index,sentiment in enumerate(df_dtm.columns):
  generate_wordcloud(df_dtm[sentiment].sort_values(ascending=False),sentiment)