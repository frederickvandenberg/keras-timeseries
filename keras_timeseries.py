##### IMPORT BASE REQUIREMENTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### ML Sentiment
df = pd.read_csv('mining_headlines_500.csv', sep=',')

from sklearn.model_selection import train_test_split

X = df['headline']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

clf = LinearSVC()
#stop_words=stopwords

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', clf),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

# Form a prediction set
predictions = text_clf.predict(X_test)

# Report the confusion matrix
from sklearn import metrics

#print(metrics.confusion_matrix(y_test,predictions))

# You can make the confusion matrix less confusing by adding labels:
dfc = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['neg','pos'], columns=['neg','pos'])
print(dfc)

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))

##### SQL IMPORT

from utils.config import config_dict
import utils.db_toolbox as tb
from datetime import datetime, timedelta

element = 'gold'
date_from = '2000-01-01'
date_to = '2020-02-01'

con = tb.db_con(config_dict)

# Headlines
df = pd.DataFrame(con.read_query(f"""select pub_date, heading, sub_heading
                                    from articles
                                    where (unique_text like '%{element}%' AND pub_date BETWEEN '{date_from}' AND '{date_to}')
                                    order by pub_date desc;"""),
                                    columns=['pub_date','heading','sub_heading'])

df['combined'] = df['heading'] + '. ' + df['sub_heading']

del df['heading']
del df['sub_heading']

# Price
pf = pd.DataFrame(con.read_query(f"""select spot_date, am_price
                                    from metal_price
                                    where (commodity like '%{element}%' AND spot_date BETWEEN '{date_from}' AND '{date_to}')
                                    order by spot_date desc;"""),
                                    columns=['spot_date','am_price'])

# Sentiment Predictions
predictions = text_clf.predict(df['combined'])

prediction_numerics = []

for i in predictions:
    if i == 'neg':
        prediction_numerics.append(-1)
    else:
        prediction_numerics.append(1)
        
# Date format
prediction_results = pd.DataFrame(predictions)
prediction_numerics = pd.DataFrame(prediction_numerics)
df['sentiment'] = prediction_results
df['num_sentiment'] = prediction_numerics
df['pub_date'] = pd.to_datetime(df['pub_date'])
pf['spot_date'] = pd.to_datetime(pf['spot_date'])

df.index = df['pub_date']
df.index = pd.to_datetime(df.index)
pf.index = pf['spot_date']
pf.index = pd.to_datetime(pf.index)

#### Time period W or M switch
time_period = 'M'

dfs = df['num_sentiment'].resample(time_period).mean().rename('Sentiment')
dff = df['num_sentiment'].resample(time_period).count().rename('Frequency')
pfp = pf['am_price'].resample(time_period).mean().rename('Price')
df1 = pd.concat([dfs,dff,pfp], axis=1)


df1.index
df1.index.freq=time_period

df1.dropna(inplace=True)
df1.drop(columns=['Sentiment'],inplace=True)

##### KERAS

test_l = 4
rangel = len(df1)-test_l
train = df1.iloc[:rangel]
test = df1.iloc[rangel:]

# scaling data, finds max value in train data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train) 

scaled_train = scaler.transform(train) # divides by max 
scaled_test = scaler.transform(test) # for test data

# keras import requirements
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# inputs
n_input = 6 
#n_features = 2

# source data, source targets is the same thus repeats
train_generator = TimeseriesGenerator(scaled_train,scaled_train,
                                length=n_input,
                                batch_size=6) #smaller batch is generally works well for RNN, too large and it overfits RNN
print(scaled_train.shape)

# model setup
model = Sequential()

model.add(LSTM(125, #number of neurons to use, play around and see
              activation='relu',
              input_shape=(n_input,scaled_train.shape[1]))) 

model.add(Dense(scaled_train.shape[1])) # directly outputs predictions
model.compile(optimizer='adam',loss='mse')

print(model.summary())

# fitting model
model.fit_generator(train_generator,epochs=50) #more epochs longer but better

# model loss
myloss = model.history.history['loss']
plt.plot(range(len(myloss)),myloss)

# reshape 
first_eval_batch = scaled_train[-n_input:]
first_eval_batch = first_eval_batch.reshape((1, n_input, scaled_train.shape[1]))

# empty place holder for test predictions
test_predictions = []
n_features = scaled_train.shape[1]
# last n_input points from training set
first_eval_batch = scaled_train[-n_input:]

#reshape to format RNN wants (same format as TimeseriesGenerator)
current_batch = first_eval_batch.reshape((1,n_input,n_features))

# len(test) can swap with 24 e.g. if want 24 months
npred = test_l
for i in range(npred):
    
    # one timestep ahed of historical 12 points
    current_pred = model.predict(current_batch)[0] #index stops it being 2D (formatting)
    
    # store that preddiction
    test_predictions.append(current_pred)
    
    # UPDATE current batch to include the prediction
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)

# gets start date of test column
a1 = test.index[0].date()

idx = pd.date_range(a1, periods=npred, freq=time_period)

true_predictions = pd.DataFrame(data=true_predictions,
                                columns=test.columns,
                               index=idx)

true_predictions.index.freq=time_period

##### PLOTTING

# Frequency prediction
plt.rcParams["figure.figsize"] = (16,8)
plt.plot(train['Frequency'], label='train') #.loc['2016-01-01':])
plt.plot(test['Frequency'],label='test')
plt.plot(true_predictions['Frequency'],label='prediction')
plt.legend(loc="lower left")
plt.title('Frequency')
plt.show()

# Price prediction
plt.plot(train['Price'],label='train') #.loc['2016-01-01':])
plt.plot(test['Price'],label='test')
plt.plot(true_predictions['Price'],label='prediction')
plt.legend(loc="upper left")
plt.title('Price')
plt.show()

##### ERROR
from statsmodels.tools.eval_measures import mse,rmse,meanabs
print(rmse(test['Price'],true_predictions['Price']))
print(rmse(test['Frequency'],true_predictions['Frequency']))

##### SAVE MODEL

model.save('keras_lstm_model.h5')