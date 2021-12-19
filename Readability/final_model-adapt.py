#!/usr/bin/env python
# coding: utf-8

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # No GPU
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'   # Stop extra messages

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from tensorflow.keras import Sequential as ksq
from tensorflow.keras import layers as kl
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy as klcc
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("D:\AML\data_graded4.csv")

data.head()

data=data[-100_000:]

data["Year"].value_counts()

spec_chars = ["1","2","3","4","5","6","7","8","9","0"]
for char in spec_chars:
    data['Abstract'] = data['Abstract'].str.replace(char, '')


# In[8]:


# 1=Flesh, 2=Smog, 3=Fog, 4=Dale, 5=Ari,6=Cl
train_col=3
model_='Fog'
model_name=model_+'_model'
model_weight=model_+'_model_weight'
fig_name=model_+'_training.png'
history_file=model_+'_history.pkl'
seed=88
epochs=20


# In[12]:


col=data.columns
data_y=data[col[1+train_col]]
#data_y=data_y*10
#data_y=np.array(data_y).astype('int')

#nc = np.unique(data_y).shape[0]


# In[13]:


# split into train test sets
X_train, X_test, Y_train, Y_test = tts(data['Abstract'], data_y, test_size=0.2,random_state=seed)
   


# In[14]:


XX_train=np.array(X_train)
XX_test=np.array(X_test)
YY_train=np.array(Y_train)
YY_test=np.array(Y_test)


# In[15]:


YY_test


# In[20]:


#Create the text encoder
VOCAB_SIZE = 5000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))


# In[21]:


vocab = np.array(encoder.get_vocabulary())
vocab[:20]

encoded_example = encoder(X_test)[:3].numpy()
encoded_example


# In[22]:


encoded_example


# In[23]:


model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[24]:


loss='sparse_categorical_crossentropy'
#loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(loss=loss, 
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[25]:


mhistory4 = model.fit(X_train,Y_train, epochs=20,
                    validation_split=0.2,
                    validation_steps=30)


# In[37]:


mhistory3 = model.fit(X_train,Y_train, epochs=20,
                    validation_split=0.2,
                    validation_steps=30)


# In[31]:


mhistory2 = model.fit(X_train,Y_train, epochs=20,
                    validation_split=0.2,
                    validation_steps=30)


# In[20]:


mhistory = model.fit(X_train,Y_train, epochs=20,
                    validation_split=0.2,
                    validation_steps=30)


# In[ ]:


print([layer.supports_masking for layer in model.layers])
sample_text = ('supply chain  demand uncertainty    challenging problem due  increased competition  market volatility  modern markets  flexibility  planning decisions makes modular manufacturing  promising way  address  problem    work   problem  multiperiod process  supply chain network design  considered  demand uncertainty   mixed integer two stage stochastic programming problem  formulated  integer variables indicating  process design  continuous variables  represent  material flow   supply chain   problem  solved using  rolling horizon approach  benders decomposition  used  reduce  computational complexity   optimization problem   promote risk averse decisions   downside risk measure  incorporated   model   results demonstrate  several advantages  modular designs  meeting product demands   pareto optimal curve  minimizing  objectives  expected cost  downside risk  obtained         american institute  chemical engineers  ')

predictions = model.predict(np.array([sample_text]))
print(predictions[0])


# In[ ]:


model.save(model_name)
model.save_weights(model_weight)
loss,acc=model.evaluate(test.batch(seed))


# In[ ]:


#%% plot                             ******************
fig=plt.figure(figsize=(10,5))
mhistory.history.keys()
epoch=range(1,len(mhistory.history['loss'])+1)
plt.subplot(1,1,1)
plt.plot(epoch,mhistory.history['loss'],'bo',label='Train_loss')
plt.plot(epoch,mhistory.history['val_loss'],'ro',label='Val_loss')
plt.plot(epoch,mhistory.history['accuracy'],'b',label='Train_accuracy')
plt.plot(epoch,mhistory.history['val_accuracy'],'r',label='Val_accuracy')
plt.legend(loc='lower left')
plt.xlabel('Epoch')
fig.savefig(fig_name)
model.evaluate(X_test,Y_test)

