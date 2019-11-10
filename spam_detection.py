# -*- coding: utf-8 -*-

"""
Created on Sat Jul 27 22:09:19 2019

@author: Habeeb"""


# coding: utf-8



#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras import metrics





data = pd.read_csv('sapam.csv',encoding='latin-1')



data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":'label', "v2":'text'})
tags = data["label"]
texts = data["text"]

for k in range(len(tags)):
    if (tags[k]=='ham'):
        tags[k]='normal'
    else:
        tags[k]='spam'

h=data['label']=='spam'
print("Spam sms count" ,len(data[h]))
h=data['label']=='normal'
print("Normal sms count" ,len(data[h]))




## For enumeration up to a maximum of 1000
num_max = 1000

## Tags make 0 and 1 mapping the tags to 1 or 0
le = LabelEncoder()
tags = le.fit_transform(tags)

## The process of enumerating words
#fit_on_texts Updates internal vocabulary based on a list of texts. 
#This method creates the vocabulary index based on word frequency.
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(texts) 

 





## A maximum of 100 words and sentences are maintained
max_len = 100
texts_seq = tok.texts_to_sequences(texts)




# A maximum of 100 words and sentences are maintained
## The number of words is made from 100. Missing words are written to 0.

texts_mat = sequence.pad_sequences(texts_seq,maxlen=max_len)






model = Sequential()
model.add(Dense(120,input_shape=(max_len,),activation='relu'))
model.add(Dense(88,activation='sigmoid'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

history=model.fit(texts_mat,tags,batch_size=175,shuffle=True,epochs=50,validation_split=0.3)

loss,accuracy= model.evaluate(texts_mat,tags)

ipstr="SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"

'''
def user_ip(ipstr):
    tok.fit_on_texts(ipstr)
    ipstr_seq = tok.texts_to_sequences(ipstr)
    ipstr_mat= sequence.pad_sequences(ipstr_seq,maxlen=max_len)
    print(model.ipstr_mat())
    '''