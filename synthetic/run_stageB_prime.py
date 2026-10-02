#!/usr/bin/env python3
"""Complete Stage B': classical, pure-QSVM, and hybrid comparisons."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load(p):
 a=pd.read_csv(p,header=None).to_numpy(); return a[:,:-1],a[:,-1].astype(int)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))
def ci_diff(y,a,b,B=2000,seed=11):
 rng=np.random.default_rng(seed); n=len(y); vals=[]
 for _ in range(B):
  ix=rng.integers(0,n,n)
  if np.unique(y[ix]).size==2: vals.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
 return np.quantile(vals,[.025,.975]).tolist()
def mcnemar(y,a,b):
 from scipy.stats import binomtest
 ac=(a==y); bc=(b==y); n01=int(np.sum(~ac&bc)); n10=int(np.sum(ac&~bc)); n=n01+n10
 return {'b':n01,'c':n10,'p_exact':1.0 if n==0 else float(binomtest(min(n01,n10),n,.5).pvalue)}
def fit_classical(X,y,seed):
 cv=StratifiedKFold(3,shuffle=True,random_state=seed)
 lr=GridSearchCV(LogisticRegression(max_iter=3000,class_weight='balanced'),{'C':[.1,1,10,100]},cv=cv,scoring='roc_auc').fit(X,y)
 rbf=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),{'C':[.1,1,10,100],'gamma':['scale',.1,1]},cv=cv,scoring='roc_auc').fit(X,y)
 return lr,rbf
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--qsvm',required=True); ap.add_argument('--out',default='stageB_prime_results.json'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--max-train',type=int,default=400,help='Stratified training cap applied identically to every model'); a=ap.parse_args()
 sys.path.insert(0,str(Path(a.qsvm).resolve().parent)); import qsvm_kernel_v2 as qk
 d=Path(a.data); sets=['in_footprint','shifted_footprint','pmu_outage']; result={'seed':a.seed,'feature_sets':{}}
 for kind in ['summary','phase']:
  X,y=load(d/f'train_{kind}.csv'); tests={s:load(d/f'{s}_{kind}.csv') for s in sets}
  if len(y)>a.max_train:
   rng=np.random.default_rng(a.seed); ix=np.concatenate([rng.choice(np.flatnonzero(y==c),a.max_train//2,replace=False) for c in [0,1]]); rng.shuffle(ix); X,y=X[ix],y[ix]
  scaler=StandardScaler().fit(X); Z=scaler.transform(X); Zt={s:(scaler.transform(v[0]),v[1]) for s,v in tests.items()}
  pca=None
  if kind=='phase':
   pca=PCA(n_components=8,random_state=a.seed).fit(Z); Z=pca.transform(Z); Zt={s:(pca.transform(v[0]),v[1]) for s,v in Zt.items()}
   # Re-standardize PCA scores using training only.
   sc2=StandardScaler().fit(Z); Z=sc2.transform(Z); Zt={s:(sc2.transform(v[0]),v[1]) for s,v in Zt.items()}
  lr,rbf=fit_classical(Z,y,a.seed)
  edges=qk.DEFAULT_EDGES; best,cvgrid=qk.cv_select(Z,y,edges,[.5,1,2],[.5,1,2],[.1,1,10],2,3,a.seed)
  alpha,alpha_auc,alpha_grid=qk.select_hybrid_alpha(Z,y,edges,best['beta'],best['beta_prime'],best['C'],2,n_folds=3,seed=a.seed)
  rr={'n_train_used_all_models':int(len(y)),'n_raw_features':int(X.shape[1]),'n_qubits':int(Z.shape[1]),'pca_explained_variance':None if pca is None else float(pca.explained_variance_ratio_.sum()),'qsvm_hyperparams':best,'hybrid_alpha':alpha,'hybrid_cv_auc':alpha_auc,'conditions':{}}
  for s,(Xt,yt) in Zt.items():
   pl=lr.predict_proba(Xt)[:,1]; pr=sigmoid(rbf.decision_function(Xt))
   _,pq,dq,align,ttr,tte,nq=qk.fit_and_eval(Z,y,Xt,best['beta'],best['beta_prime'],edges,best['C'],2,None)
   _,ph,dh,_,htr,hte,_=qk.fit_and_eval(Z,y,Xt,best['beta'],best['beta_prime'],edges,best['C'],2,alpha)
   preds={'LR':(pl,pl>=.5),'RBF':(pr,pr>=.5),'QSVM':(pq,dq),'Hybrid':(ph,dh)}; metrics={}
   for name,(score,pred) in preds.items(): metrics[name]={'auc':float(roc_auc_score(yt,score)),'mcnemar_vs_LR':mcnemar(yt,pred.astype(int),(pl>=.5).astype(int))}
   metrics['QSVM']['auc_difference_ci_vs_LR']=ci_diff(yt,pq,pl); metrics['QSVM']['auc_difference_ci_vs_RBF']=ci_diff(yt,pq,pr)
   metrics['Hybrid']['auc_difference_ci_vs_LR']=ci_diff(yt,ph,pl); metrics['Hybrid']['auc_difference_ci_vs_RBF']=ci_diff(yt,ph,pr)
   rr['conditions'][s]={'metrics':metrics,'kernel_alignment':align,'timing_seconds':{'qsvm_train':ttr,'qsvm_test':tte,'hybrid_train':htr,'hybrid_test':hte}}
  result['feature_sets'][kind]=rr
 Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
