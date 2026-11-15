#!/usr/bin/env python3
import csv, os
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib

FEATURE_NAMES = [
    "MI_dir_L5_weight","MI_dir_L5_mean","MI_dir_L5_variance",
    "MI_dir_L3_weight","MI_dir_L3_mean","MI_dir_L3_variance",
    "MI_dir_L1_weight","MI_dir_L1_mean","MI_dir_L1_variance",
    "MI_dir_L0.1_weight","MI_dir_L0.1_mean","MI_dir_L0.1_variance",
    "MI_dir_L0.01_weight","MI_dir_L0.01_mean","MI_dir_L0.01_variance",
    "H_L5_weight","H_L5_mean","H_L5_variance",
    "H_L3_weight","H_L3_mean","H_L3_variance",
    "H_L1_weight","H_L1_mean","H_L1_variance",
    "H_L0.1_weight","H_L0.1_mean","H_L0.1_variance",
    "H_L0.01_weight","H_L0.01_mean","H_L0.01_variance",
    "HH_L5_weight","HH_L5_mean","HH_L5_std","HH_L5_magnitude","HH_L5_radius","HH_L5_covariance","HH_L5_pcc",
    "HH_L3_weight","HH_L3_mean","HH_L3_std","HH_L3_magnitude","HH_L3_radius","HH_L3_covariance","HH_L3_pcc",
    "HH_L1_weight","HH_L1_mean","HH_L1_std","HH_L1_magnitude","HH_L1_radius","HH_L1_covariance","HH_L1_pcc",
    "HH_L0.1_weight","HH_L0.1_mean","HH_L0.1_std","HH_L0.1_magnitude","HH_L0.1_radius","HH_L0.1_covariance","HH_L0.1_pcc",
    "HH_L0.01_weight","HH_L0.01_mean","HH_L0.01_std","HH_L0.01_magnitude","HH_L0.01_radius","HH_L0.01_covariance","HH_L0.01_pcc",
    "HH_jit_L5_weight","HH_jit_L5_mean","HH_jit_L5_variance",
    "HH_jit_L3_weight","HH_jit_L3_mean","HH_jit_L3_variance",
    "HH_jit_L1_weight","HH_jit_L1_mean","HH_jit_L1_variance",
    "HH_jit_L0.1_weight","HH_jit_L0.1_mean","HH_jit_L0.1_variance",
    "HH_jit_L0.01_weight","HH_jit_L0.01_mean","HH_jit_L0.01_variance",
    "HpHp_L5_weight","HpHp_L5_mean","HpHp_L5_std","HpHp_L5_magnitude","HpHp_L5_radius","HpHp_L5_covariance","HpHp_L5_pcc",
    "HpHp_L3_weight","HpHp_L3_mean","HpHp_L3_std","HpHp_L3_magnitude","HpHp_L3_radius","HpHp_L3_covariance","HpHp_L3_pcc",
    "HpHp_L1_weight","HpHp_L1_mean","HpHp_L1_std","HpHp_L1_magnitude","HpHp_L1_radius","HpHp_L1_covariance","HpHp_L1_pcc",
    "HpHp_L0.1_weight","HpHp_L0.1_mean","HpHp_L0.1_std","HpHp_L0.1_magnitude","HpHp_L0.1_radius","HpHp_L0.1_covariance","HpHp_L0.1_pcc",
    "HpHp_L0.01_weight","HpHp_L0.01_mean","HpHp_L0.01_std","HpHp_L0.01_magnitude","HpHp_L0.01_radius","HpHp_L0.01_covariance","HpHp_L0.01_pcc",
]
OUT_DIR="/work/output"
def load(d,k):
    p=os.path.join(OUT_DIR,f'device{d}_{k}_capture_features.csv')
    df=pd.read_csv(p); X=df[FEATURE_NAMES].values.astype('float32')
    return np.nan_to_num(X,0,0,0)
def mse(m,s,X): Xs=s.transform(X); Xr=m.predict(Xs,verbose=0); return np.mean((Xs-Xr)**2,axis=1)
def thr(e1): med=np.median(e1); mad=np.median(np.abs(e1-med)); return med+2.5*mad
def stats(yt,yp):
    tp=sum(1 for t,pv in zip(yt,yp) if t==1 and pv==1); fp=sum(1 for t,pv in zip(yt,yp) if t==0 and pv==1)
    tn=sum(1 for t,pv in zip(yt,yp) if t==0 and pv==0); fn=sum(1 for t,pv in zip(yt,yp) if t==1 and pv==0)
    acc=(tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0; prec=tp/(tp+fp) if (tp+fp) else 0; rec=tp/(tp+fn) if (tp+fn) else 0; f1=2*prec*rec/(prec+rec) if (prec+rec) else 0
    return tp,fp,tn,fn,acc,prec,rec,f1
cands=[('C1','/work/new_models/c1/autoencoder_allinone.keras','/work/new_models/c1/scaler_allinone.pkl'),('C2','/work/model/autoencoder_allinone.keras','/work/model/scaler_allinone.pkl'),('C3','/work/new_models/c3/autoencoder_allinone.keras','/work/new_models/c3/scaler_allinone.pkl')]
res=[]
for cand,mk,sk in cands:
    model=keras.models.load_model(mk,compile=False); scaler=joblib.load(sk)
    for d in range(1,10):
        Xr1=load(d,'normal_r1'); Xr2=load(d,'normal_r2'); Xar=load(d,'attack'); Xrh=load(d,'attack_hard')
        e1=mse(model,scaler,Xr1); t=thr(e1)
        er2=mse(model,scaler,Xr2); ear=mse(model,scaler,Xar); erh=mse(model,scaler,Xrh)
        yt=[0]*len(er2)+[1]*len(ear); yp=[int(x>t) for x in er2]+[int(x>t) for x in ear]
        tp,fp,tn,fn,acc,prec,rec,f1 = stats(yt,yp)
        yt_h=[0]*len(er2)+[1]*len(erh); yp_h=[int(x>t) for x in er2]+[int(x>t) for x in erh]
        tp_h,fp_h,tn_h,fn_h,acc_h,prec_h,rec_h,f1_h = stats(yt_h,yp_h)
        fpr_r2=float(np.sum(er2>t))/len(er2) if len(er2) else 0
        res.append({'candidate':cand,'device':str(d),'threshold':t,'fpr_r2':fpr_r2,'tpr_attack':rec,'precision_attack':prec,'recall_attack':rec,'f1_attack':f1,'accuracy_attack':acc,'tp_attack':tp,'fp_attack':fp,'tn_attack':tn,'fn_attack':fn,'tpr_attack_hard':rec_h,'f1_attack_hard':f1_h,'tp_attack_hard':tp_h,'fp_attack_hard':fp_h,'tn_attack_hard':tn_h,'fn_attack_hard':fn_h})
with open('/work/output/candidates_current.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=res[0].keys()); w.writeheader(); w.writerows(res)
print('done')
