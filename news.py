#!/usr/bin/env python
# coding: utf-8

# In[5]:


# %sc !wget 'https://s3.amazonaws.com/sagemakermathan/bitstamp.csv'
# %sc !wget 'https://s3.amazonaws.com/sagemakermathan/coinbase.csv'
# %sc !wget 'https://s3.amazonaws.com/sagemakermathan/cryptonewskaggle.csv'
# %sc !wget 'https://s3.amazonaws.com/sagemakermathan/news.csv'
# %sc !wget 'https://s3.amazonaws.com/sagemakermathan/cryptonewskagglecleaned.csv'


# In[64]:


# import pandas as pd
# coinbase=pd.read_csv('coinbase.csv')
# cryptonewsonly=pd.read_csv('cryptonewscleanedallmagic.csv',encoding='iso-8859-1')


# In[8]:


import pandas as pd
import numpy as np
import OpenBlender
import json
def news():
    token = '625158b795162979550da93cIRNBlHXKt6T1MNbtQ3wsLyUwIqDhF1'
    action = 'API_getObservationsFromDataset'
    # ANCHOR: 'Bitcoin vs USD'

    parameters = {
        'token' : token,
        'id_dataset' : '5d4c3af79516290b01c83f51',
        'date_filter':{"start_date" : "2020-01-01",
                       "end_date" : "2021-01-09"}
    }
    df = pd.read_json(json.dumps(OpenBlender.call(action, parameters)['sample']), convert_dates=False, convert_axes=False).sort_values('timestamp', ascending=False)
    df.reset_index(drop=True, inplace=True)
    df['date'] = [OpenBlender.unixToDate(ts, timezone = 'GMT') for ts in df.timestamp]
    df = df.drop('timestamp', axis = 1)


    # In[9]:


    # get_ipython().system('pip install openblender')


    # In[10]:


    # print(df.shape)


    # In[11]:


    # df.head


    # In[12]:


    df['log_diff']=np.log(df['price'])-np.log(df['open'])
    # df


    # In[13]:


    import plotly.offline as py
    import plotly.graph_objs as go

    # price = go.Scatter(x=df['date'], y=df['price'], name= 'Price')
    # py.iplot([price])


    #

    # hello = go.Scatter(x=df['date'], y=df['log_diff'], name= 'Price')
    # py.iplot([hello])

    # In[14]:


    df['target']=[1 if log_diff>0 else 0 for log_diff in df['log_diff']]
    # df


    # In[15]:


    format = '%d-%m-%Y %H:%M:%S'
    timezone = 'GMT'
    df['timestamp'] = OpenBlender.dateToUnix(df['date'],
                                               date_format = format,
                                               timezone = timezone)
    df = df[['date','timestamp', 'price', 'target']]
    # df.head()


    # In[16]:


    key = "bitcoin"
    df=df.sort_values('timestamp').reset_index(drop=True)
    # print('From : ' + OpenBlender.unixToDate(min(df.timestamp)))
    # print('Until: ' + OpenBlender.unixToDate(max(df.timestamp)))
    OpenBlender.searchTimeBlends(token,df.timestamp,key)


    # In[17]:


    blend_source = {
                    'id_dataset':'5ea2039095162936337156c9',
                    'feature' : 'text'
                }

    df_blend = OpenBlender.timeBlend( token = token,
                                      anchor_ts = df.timestamp,
                                      blend_source = blend_source,
                                      blend_type = 'agg_in_intervals',
                                      interval_size = 60 * 60 * 24,
                                      direction = 'time_prior',
                                      interval_output = 'list',
                                      missing_values = 'raw')
    df = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)
    # df.head()


    # In[18]:


    a=df['BITCOIN_NE.text_last1days'][:1]
    b=df['BITCOIN_NE.text_last1days'][1:2]
    print(a.to_string())
    return a.to_string(),b.to_string()

# In[19]:

'''
positive_filter = {'name' : 'positive', 
                   'match_ngrams': ['positive', 'buy', 
                                    'bull', 'boost','bullish','growth','increase','peak','high','profits','upmarket','booming','roaring trade','unicorn','booming trade','surplus','up']}
blend_source = {
                'id_dataset':'5ea2039095162936337156c9',
                'feature' : 'text',
                'filter_text' : positive_filter
            }
df_blend = OpenBlender.timeBlend( token = token,
                                  anchor_ts = df.timestamp,
                                  blend_source = blend_source,
                                  blend_type = 'agg_in_intervals',
                                  interval_size = 60 * 60 * 24,
                                  direction = 'time_prior',
                                  interval_output = 'list',
                                  missing_values = 'raw')
df = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)

negative_filter = {'name':'negative','match_ngrams': ['crash','decrease','low','depression','loss','deficit','down','drop','downfall','bear','bearish']}
blend_source = {
                'id_dataset':'5ea2039095162936337156c9',
                'feature' : 'text',
                'filter_text' : negative_filter
            }
df_blend = OpenBlender.timeBlend( token = token,
                                  anchor_ts = df.timestamp,
                                  blend_source = blend_source,
                                  blend_type = 'agg_in_intervals', #closest_observation
                                  interval_size = 60 * 60 * 24,
                                  direction = 'time_prior',
                                  interval_output = 'list',
                                  missing_values = 'raw')
df = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)


# In[20]:


df


# In[21]:


# import matplotlib.pyplot as plt
# import numpy as np

# xpoints = date
# ypoints = BITCOIN_NE.text_COUNT_last1days:positive

# plt.plot(xpoints, ypoints)
# plt.show()
import plotly.offline as py
import plotly.graph_objs as go

price = go.Scatter(x=df['date'], y=df['BITCOIN_NE.text_COUNT_last1days'], name= 'Price')
# py.iplot([price])


# In[79]:


features = ['target', 'BITCOIN_NE.text_COUNT_last1days:positive', 'BITCOIN_NE.text_COUNT_last1days:negative']
df[features].corr()['target']


# In[22]:


blend_source = { 
                'id_textVectorizer':'5f739fe7951629649472e167'
               }
df_blend = OpenBlender.timeBlend( token = token,
                                  anchor_ts = df.timestamp,
                                  blend_source = blend_source,
                                  blend_type = 'agg_in_intervals',
                                  interval_size = 60 * 60 * 24,
                                  direction = 'time_prior',
                                  interval_output = 'list',
                                  missing_values = 'raw') .add_prefix('VEC.')
df = pd.concat([df, df_blend.loc[:, df_blend.columns != 'timestamp']], axis = 1)
df.head()
df.to_csv("senti.csv")


# In[81]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import sklearn.metrics as metrics
# We drop correlated features because with so many binary 
# ngram variables there's a lot of noise
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
df.drop([column for column in upper.columns if any(upper[column] > 0.5)], axis=1, inplace=True)

# Now we separate in train/test sets
df_=df
# X= df_.drop('target',axis=1).values
X = df.select_dtypes(include=[np.number]).drop('target', axis = 1).values
y = df.loc[:,['target']].values
div = int(round(len(X) * 0.2))
X_train = X[:div]
y_train = y[:div]
X_test = X[div:]
y_test = y[div:]
# Finally, we perform ML and see results
rf = RandomForestRegressor(n_estimators = 1000, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
df_res = pd.DataFrame({'y_test':y_test[:, 0], 'y_pred':y_pred})
threshold = 0.5
preds = [1 if val > threshold else 0 for val in df_res['y_pred']]
print(metrics.confusion_matrix(preds, df_res['y_test']))
print('Accuracy Score:')
print(accuracy_score(preds, df_res['y_test']))
print('Precision Score:')
print(precision_score(preds, df_res['y_test']))


# In[1]:


# get_ipython().system('pip install monkeylearn')


# In[4]:


# from monkeylearn import MonkeyLearn
# ml = MonkeyLearn('1d3386cf4b05a1940b36d3dc0b40a6510cdbdeec')


# In[5]:


# print(ml)


# In[ ]:
'''



