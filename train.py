"""
training models
"""
import pandas as pd
import numpy as np
import dispatcher
import matplotlib.pyplot as plt
from sklearn import preprocessing,metrics,feature_selection
import warnings
import pickle
warnings.filterwarnings('ignore')

def run(fold,model):
    #fold=2
    #model='xgb' 
    df=pd.read_csv('../inputs/train_preprocessed.csv').drop('ID',axis=1)
    
    df_train=df[df.kfold!=fold].reset_index(drop=True)
    df_valid=df[df.kfold==fold].reset_index(drop=True)

    features=[f for f in df.columns if f not in ['Segmentation','kfold']] #'Var_1','Profession','Gender', 'Ever_Married',
    x_train=df_train[features]
    y_train=df_train.Segmentation.values
    x_valid=df_valid[features]
    y_valid=df_valid.Segmentation.values
    
    #Scalimg
    sc=preprocessing.StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_valid=sc.transform(x_valid)
    #initialte model
    model=dispatcher.models[model]
    
    #Selecting Best Features
    sfm=feature_selection.SelectFromModel(estimator=model)
    x_transformed=sfm.fit_transform(x_train,y_train)
    feats=sfm.get_support(indices=True)
    
    #fitting the model
    model.fit(x_transformed,y_train)
   
    # Plotting of Feature Importance
# =============================================================================
#     col_name=features
#     imp=model.feature_importances_
#     indx=np.argsort(imp)
#     plt.title(f'{model} Feature Importance')
#     plt.barh(range(len(indx)), imp[indx])
#     plt.yticks(range(len(indx)),[col_name[i] for i in indx])
#     plt.xlabel('Feature Importances')
#     plt.show()
# =============================================================================
    
 
    #pickle.dump(model, open(f"../models/{mod}_{folds}.pickle.dat",'wb')) #.dat for XGBoost
    #prediction
    y_pred=model.predict(x_valid[:,feats])
    #ROC AUC Score
    acc=metrics.accuracy_score(y_valid, y_pred)
    
    print(f'Fold= {folds} and Accuracy= {acc}')
    
    
    
if __name__=='__main__':
    model=['lgb','rf','xgb']
    for mod in model:
        print(mod)
        for folds in range(5):
            run(folds,mod)