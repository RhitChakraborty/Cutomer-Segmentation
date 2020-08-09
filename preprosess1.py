
"""
Created on Fri Aug  7 22:14:17 2020

@author: rhitc
"""

import pandas as pd
import numpy as np
from sklearn import model_selection,preprocessing


if __name__=="__main__":
    df_train=pd.read_csv('../inputs/train.csv')
    df_test=pd.read_csv('../inputs/test.csv')
    
    df=pd.concat([df_train,df_test]).reset_index(drop=True)
    
    df.Ever_Married=df.Ever_Married.fillna('None')
    df.Profession=df.Profession.fillna('None')
    df.Var_1=df.Var_1.fillna('None')
    df.Work_Experience=df.Work_Experience.fillna(1)
    df.Family_Size=df.Family_Size.fillna(1)
    
    df.loc[(df.Profession=='Doctor'),'Graduated']=df.loc[(df.Profession=='Doctor'),'Graduated'].fillna('Yes')
    df.loc[(df.Profession=='Engineer'),'Graduated']=df.loc[(df.Profession=='Engineer'),'Graduated'].fillna('Yes')
    df.loc[(df.Profession=='Lawyer'),'Graduated']=df.loc[(df.Profession=='Lawyer'),'Graduated'].fillna('Yes')
    df.Graduated=df.Graduated.fillna('None')
    
    
    df['Prof+grad']=df.Profession+'_'+df.Graduated
    
    #label encoding
    map_ss={
        'Average':1, 
        'Low':0, 
        'High':2
        }
     
    map_seg={
         'A':0,
         'B':1,
         'C':2,
         'D':3
         }
    #ohe Spending score
    df_ss=pd.get_dummies(df['Spending_Score'],prefix='Spending_Score',prefix_sep=':')
    df=pd.concat([df,df_ss],axis=1)
    #label Encoding Spending score
    df.Spending_Score=df.Spending_Score.map(map_ss)
    #mapping Segmentation
    df.Segmentation=df.Segmentation.map(map_seg)
    
    features=['Gender','Ever_Married','Graduated', 'Profession','Var_1','Prof+grad']
    new_df=pd.DataFrame()
    for col in features:
        le=preprocessing.LabelEncoder()
        le.fit(df[col].values)
        df.loc[:,col]=le.transform(df[col].values)
        ohe=preprocessing.OneHotEncoder()
        arr=ohe.fit_transform(df[[col]]).toarray()
        labels=[col+':'+str(label) for label in le.classes_]
        
        temp=pd.DataFrame(arr,columns=labels)
        new_df=pd.concat([new_df,temp],axis=1)
        
    
    df=pd.concat([df,new_df],axis=1)
    
    #Feature Creation
    temp_df=df.groupby(['Age']).agg({
        'Spending_Score':['count','mean','sum'],
        'Work_Experience':['count','sum','mean'],
        'Graduated':['count'],
        'Ever_Married':['count'],
        'Gender':['count'], 
        'Family_Size':['count','sum','min','max'],
        'Age':['count'],
        'Var_1':['count']
        })
    temp_df.columns=['_'.join(x) for x in temp_df.columns]
    
    df=pd.merge(df,temp_df,on='Age',how='left')
    
    train_df=df[df.Segmentation.notnull()]
    test_df=df[df.Segmentation.isnull()]
    
    #StratifiedK-fold
    train_df=train_df.sample(frac=1).reset_index(drop=True)
    y=train_df.Segmentation
    train_df['kfold']=-1
    kf=model_selection.StratifiedKFold(n_splits=5)

    for f,(trn_,val_) in enumerate(kf.split(X=train_df, y=y)):
        train_df.loc[val_,'kfold']=f
    
    #saving all preprocessed Data
    df.to_csv("../inputs/full_preprocessed.csv",index=False)
    test_df.to_csv("../inputs/test_preprocessed.csv",index=False)
    train_df.to_csv("../inputs/train_preprocessed.csv",index=False)
    
    
    
        
    
    
    
    
    