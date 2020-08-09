"""
TensorFLow
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics,preprocessing
from keras.utils import np_utils

def run():
    #fold=3
    df=pd.read_csv('../inputs/train_folds.csv')
     
    df_train=df
    #df_train=df[df.kfold!=fold].reset_index(drop=True)
    #df_valid=df[df.kfold==fold].reset_index(drop=True)
    
    sc=preprocessing.StandardScaler()
    df_train.loc[:,['Age','Work_Experience','Family_Size']]=sc.fit_transform(df_train.loc[:,['Age','Work_Experience','Family_Size']].values)
    #df_valid.loc[:,['Age','Work_Experience','Family_Size']]=sc.fit_transform(df_valid.loc[:,['Age','Work_Experience','Family_Size']].values)
    
    features=[f for f in df.columns if f not in ['ID','Segmentation','kfold']] #'Var_1','Profession','Gender', 'Ever_Married',
    x_train=df_train[features].values
    y_train=df_train.Segmentation.values.reshape(-1,1)
    #x_valid=df_valid[features].values
    #y_valid=df_valid.Segmentation.values
    
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(y_train)

    ann=tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=32,activation='relu'))
    ann.add(tf.keras.layers.Dropout(0.20))
    ann.add(tf.keras.layers.Dense(units=64,activation='relu'))
    ann.add(tf.keras.layers.Dropout(0.20))
    ann.add(tf.keras.layers.Dense(units=64,activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=16,activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=4,activation='softmax'))
    
    ann.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    ann.fit(x_train,dummy_y,batch_size=32,epochs=500)
  


    test=pd.read_csv('../inputs/test_folds.csv').drop('ID',axis=1)
    sample=pd.read_csv('../inputs/sample_submission.csv')
    
    sc=preprocessing.StandardScaler()
    test.loc[:,['Age','Work_Experience','Family_Size']]=sc.fit_transform(test.loc[:,['Age','Work_Experience','Family_Size']].values)

    pred=ann.predict(test.values)

    pred=np.argmax(pred,axis=-1)
    pred=pd.DataFrame(pred)
    mapping={
         0:'A',
         1:'B',
         2:'C',
         3:'D'
     }
    pred=pred[0].map(mapping)
    sample['Segmentation']=pred
    
    sample.to_csv("../inputs/sub_tf.csv",index=False)
    
if __name__=="__main__":
    run() 
