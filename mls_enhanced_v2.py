#!/usr/bin/env python3
# Optimized MLS Wins Prediction Script - Ridge (poly2), RF, XGB, Ensemble
import os

import warnings, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize
from packaging import version
import xgboost as xgb_mod


def safe_div(a,b,eps=1e-10): return a/(b+eps)
def evaluate(y,preds,name):
    mae=mean_absolute_error(y,preds); rmse=np.sqrt(mean_squared_error(y,preds)); r2=r2_score(y,preds)
    print(f"{name:25} -> MAE:{mae:.4f}, RMSE:{rmse:.4f}, R2:{r2:.4f}")
    return mae,rmse,r2

# Config

TRAIN_PATH = os.getenv("TRAIN_CSV","/home/gjh/dsaicourse/mlskaggle/data/train.csv")
TEST_PATH = os.getenv("TEST_CSV", "/home/gjh/dsaicourse/mlskaggle/data/test.csv")
OUT_DIR = os.getenv("OUTPUT_DIR", "/home/gjh/dsaicourse/mlskaggle/submissions/")
RS=42

print("TRAIN_PATH:",TRAIN_PATH)

RIDGE_CSV=os.path.join(OUT_DIR,"ridge_predictions.csv")
RF_CSV=os.path.join(OUT_DIR,"rf_predictions.csv")
XGB_CSV=os.path.join(OUT_DIR,"xgb_predictions.csv")
ENS_CSV=os.path.join(OUT_DIR,"ensemble_predictions.csv")

print("Loading data...")
train=pd.read_csv(TRAIN_PATH)
test=pd.read_csv(TEST_PATH)

def engineer(df):
    df=df.copy()
    for c in ['R','RA','G','SV','ERA','OBP','OPS','AB','H','2B','3B','HR','BB','SO']:
        if c not in df: df[c]=0.0
    df['R_diff_per_game']=safe_div(df['R']-df['RA'],df['G'])
    df['Save_ratio']=safe_div(df['SV'],df['G'])
    df['ERA_inverse']=safe_div(1.0,df['ERA']+1e-10)
    df['OBP_minus_RA']=df['OBP']-safe_div(df['RA'],df['G'])
    df['OPS_plus']=safe_div(df['OPS'],df['OPS'].mean() or 1)*100
    return df

train,test=engineer(train),engineer(test)
drop_cols=['W','ID','team','teamID','season','year_label','decade_label','win_bins']
num_train=train.drop(columns=[c for c in drop_cols if c in train],errors='ignore').select_dtypes(include=[np.number])
num_test=test.drop(columns=[c for c in drop_cols if c in test],errors='ignore').select_dtypes(include=[np.number])
common=[c for c in num_train.columns if c in num_test.columns]
X,y,X_test=num_train[common],train['W'],num_test[common]
corr=X.corrwith(y).abs().sort_values(ascending=False); feats=corr.head(min(30,len(corr))).index.tolist()
X,X_test=X[feats],X_test[feats]
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=RS)

sc=StandardScaler().fit(X_train)
X_train_s,X_val_s,X_s,X_test_s=sc.transform(X_train),sc.transform(X_val),sc.transform(X),sc.transform(X_test)

poly=PolynomialFeatures(2,include_bias=False)
Xtr_p,Xv_p=poly.fit_transform(X_train_s),poly.transform(X_val_s)
ridge=RidgeCV(alphas=np.logspace(-3,3,20),cv=5).fit(Xtr_p,y_train)
r_pred=ridge.predict(Xv_p); evaluate(y_val,r_pred,"Ridge (poly2)")

rf_gs=GridSearchCV(RandomForestRegressor(random_state=RS,n_jobs=-1),
                   {'n_estimators':[400,600],'max_depth':[12,16],'min_samples_split':[2,5],
                    'min_samples_leaf':[1,2],'max_features':['sqrt']},
                   cv=3,scoring='neg_mean_absolute_error',n_jobs=-1)
rf_gs.fit(X_train,y_train); rf=rf_gs.best_estimator_
rf_pred=rf.predict(X_val); evaluate(y_val,rf_pred,"Random Forest (best)")

xgb=XGBRegressor(n_estimators=1200,learning_rate=0.02,max_depth=6,subsample=0.85,
                 colsample_bytree=0.85,reg_alpha=0.4,reg_lambda=1.2,random_state=RS,n_jobs=-1,verbosity=0)

xgb_version = xgb_mod.__version__

try:
    if version.parse(xgb_version) >= version.parse("2.0.0"):
        # XGBoost ≥ 2.0 uses callbacks
        from xgboost.callback import EarlyStopping
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[EarlyStopping(rounds=50, save_best=True)],
            verbose=False
        )
    else:
        # XGBoost < 2.0 uses early_stopping_rounds
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
except TypeError:
    # Fallback in case neither argument is accepted
    xgb.fit(X_train, y_train)

xgb_pred=xgb.predict(X_val); evaluate(y_val,xgb_pred,"XGBoost")


stack=np.vstack([r_pred,rf_pred,xgb_pred]).T
def mae_loss(w): w=np.abs(w); w=w/w.sum(); return mean_absolute_error(y_val,stack.dot(w))
res=minimize(mae_loss,[1,1,1],bounds=[(0,1)]*3,constraints={'type':'eq','fun':lambda w:np.sum(w)-1})
wts=res.x if res.success else np.array([1,1,1])/3; print("Opt ensemble weights:",wts)
ens_pred=stack.dot(wts); evaluate(y_val,ens_pred,"Ensemble (opt)")

sc_full=StandardScaler().fit(X)
Xf_s,Xt_s=sc_full.transform(X),sc_full.transform(X_test)
poly_full=PolynomialFeatures(2,include_bias=False).fit(Xf_s)
Xf_p,Xt_p=poly_full.transform(Xf_s),poly_full.transform(Xt_s)
ridge_full=RidgeCV(alphas=np.logspace(-3,3,20),cv=5).fit(Xf_p,y)
rf_full=rf.__class__(**rf.get_params()).fit(X,y)
xgb_full=XGBRegressor(n_estimators=1200,learning_rate=0.02,max_depth=6,subsample=0.85,colsample_bytree=0.85,
                      reg_alpha=0.4,reg_lambda=1.2,random_state=RS,n_jobs=-1,verbosity=0).fit(X,y,verbose=False)

ridge_out=np.rint(ridge_full.predict(Xt_p)).astype(int)
rf_out=np.rint(rf_full.predict(X_test)).astype(int)
xgb_out=np.rint(xgb_full.predict(X_test)).astype(int)
ens_out=np.rint(np.vstack([ridge_out,rf_out,xgb_out]).T.dot(wts)).astype(int)

ids=test['ID'] if 'ID' in test else range(len(test))
for arr,path in [(ridge_out,RIDGE_CSV),(rf_out,RF_CSV),(xgb_out,XGB_CSV),(ens_out,ENS_CSV)]:
    pd.DataFrame({'ID':ids,'W':np.clip(arr,0,None)}).to_csv(path,index=False)
    print("Wrote",path)
print("\nAll done.")