#!/usr/bin/env python
# coding: utf-8

# # Problem Statement - A person wants to know customer sentiments about    Fortune Soyabean Oil on Amazon
# Steps -
#     1-Web Scrapping of Amazon first 3 pages of Reviews for Fortune Oil
#     2-Using pretrained NLTK Sentiment Analyser to tag sentiments
#     3-Pie Chart using Matplotlib
#     4- Wordcloud to visualize freq of words
#     5-Very small size data used but we get a holistic view of customer sentiments about this product

# # Findings :
#     
#     Major Problem detected is leakeage of oil
#     70% customers are happy with the product but 30% have issues or are nearly neutral in reviews 
#     30% Customers may switch to other Oils 
#     Leakeage problem needs to be addressed(It may be due to transportation mishandling,packaging issues )

# In[ ]:


# importing all necessary modules 
from bs4 import BeautifulSoup
import pandas as pd
import urllib


# In[ ]:


url="https://www.amazon.in/Fortune-Soyabean-Oil-1L-Pouch/product-reviews/B00TX50T4K/ref=cm_cr_othr_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber"


# In[ ]:


url1="https://www.amazon.in/Fortune-Soyabean-Oil-1L-Pouch/product-reviews/B00TX50T4K/ref=cm_cr_othr_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber-1"
url2="https://www.amazon.in/Fortune-Soyabean-Oil-1L-Pouch/product-reviews/B00TX50T4K/ref=cm_cr_othr_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber-2"
url3="https://www.amazon.in.in/Fortune-Soyabean-Oil-1L-Pouch/product-reviews/B00TX50T4K/ref=cm_cr_othr_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber-3"


# In[ ]:


data1=urllib.request.urlopen(url1).read()
data2=urllib.request.urlopen(url2).read()
data3=urllib.request.urlopen(url3).read()


# In[ ]:


data=data1+data2+data3


# In[92]:


reviews=BeautifulSoup(data,'html.parser')
con= reviews.find_all('div',class_="a-row a-spacing-small review-data")


# In[93]:


comments=[]
for i in con:
        review=i.find('span').text
        comments.append(review)


# In[94]:


comments=[w.replace('\n','')for w in comments]


# In[95]:


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatize=WordNetLemmatizer()
from nltk.corpus import stopwords
words=stopwords.words('english')
import re


# In[96]:


cleaned_reviews=[]
for review in comments:
    review= re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|\...|([0-9]+)"," ",review)
    review=word_tokenize(review)
    review=[w for w in review if w.lower() not in words]
    review=[lemmatize.lemmatize(w) for w in review]
    review=" ".join(review)
    cleaned_reviews.append(review)


# In[97]:


cleaned_reviews[0:5]


# In[98]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()


# In[99]:


df=pd.DataFrame(cleaned_reviews,columns=['review'])
df['score']=df['review'].apply(lambda x : sia.polarity_scores(x))
df['compound']=df['score'].apply(lambda x:x['compound'])
df['sentiment']=df['compound'].apply(lambda x : 'pos'if x>0 else 'neg')


# In[100]:


df.head()


# In[102]:


df.info()


# In[101]:


sentiment_dataframe=df.drop(['score','compound'],axis=1)
positive=sentiment_dataframe['sentiment'].value_counts()['pos']
negative=sentiment_dataframe['sentiment'].value_counts()['neg']


# In[104]:


#Creating pie chart
labels = 'Positive', 'Negative'
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie([positive,negative], explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True,colors=['blue','green'], startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[105]:


#Filtering records with negative tags from dataframe
negative_data=sentiment_dataframe[(sentiment_dataframe.sentiment=='neg')]
negative_data.reset_index()
negative_data


# In[106]:


''.join(negative_data['review'].values)


# In[107]:


# importing all necessary modules for wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)


# In[108]:


wordcloud = WordCloud(width = 800, height = 800,
				background_color ='white',
				stopwords = stopwords,
				min_font_size = 10).generate(''.join(negative_data['review'].values))

# plot the WordCloud image					
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)


# In[109]:


from nltk import word_tokenize
tokens=word_tokenize(''.join(negative_data['review'].values))


# In[110]:


fd=nltk.FreqDist(tokens)


# In[111]:


fd.most_common(10)


# In[112]:


fd.tabulate()


# In[113]:


fd.plot(cumulative=True)


# In[114]:


fd.plot()

