#!/usr/bin/env python3
"""Nested event-held-out static/dynamic probability ensembles for ISO-NE."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import run_isone_semisynthetic_validation as base
import run_isone_nonstationary_experiments as ns

ALPHAS=[0,.25,.5,.75,1]

def stack(events,kind):
    X=np.vstack([e[0][kind] for e in events]); y=np.hstack([e[1] for e in events]); g=np.hstack([np.full(len(e[1]),i) for i,e in enumerate(events)])
    return X,y,g

def classical_predict(events,Xt,kind,model):
    X,y,g=stack(events,kind); nq=min(ns.NQ,X.shape[1]); cv=list(LeaveOneGroupOut().split(X,y,g)); prep=[('s',StandardScaler())]
    if X.shape[1]>nq: prep.append(('pca',PCA(n_components=nq)))
    if model=='LR':
        est=Pipeline(prep+[('m',LogisticRegression(max_iter=3000,class_weight='balanced'))]); grid={'m__C':[.1,1,10]}
    else:
        est=Pipeline(prep+[('m',SVC(kernel='rbf',class_weight='balanced'))]); grid={'m__C':[.1,1,10],'m__gamma':['scale',.1,1]}
    fit=GridSearchCV(est,grid,cv=cv,scoring='roc_auc').fit(X,y)
    p=fit.predict_proba(Xt)[:,1] if model=='LR' else base.sigmoid(fit.decision_function(Xt))
    return p,fit.best_params_

def qsvm_predict(events,Xt,kind,qk):
    X,y,g=stack(events,kind); nq=min(ns.NQ,X.shape[1]); best,edges=ns.qselect(X,y,g,kind,qk); tf,Z=ns.fit_transform(X,nq); Zt=ns.apply_transform(Xt,tf)
    _,p,_,_,_,_,_=qk.fit_and_eval(Z,y,Zt,best['beta'],best['beta_prime'],edges,best['C'],2,None)
    return p,best

def predict(events,Xt,kind,model,qk):
    return qsvm_predict(events,Xt,kind,qk) if model=='QSVM' else classical_predict(events,Xt,kind,model)

def ensemble_fold(events,test,model,qk):
    # Inner event-held-out predictions. Each expert is retuned using only the
    # remaining events before predicting the inner validation event.
    held=[]
    for j in range(len(events)):
        tr=[events[i] for i in range(len(events)) if i!=j]; va=events[j]
        ps,_=predict(tr,va[0]['static'],'static',model,qk); pd,_=predict(tr,va[0]['structured'],'structured',model,qk)
        held.append((va[1],ps,pd))
    cvrows=[]
    for a in ALPHAS:
        auc=[roc_auc_score(y,a*ps+(1-a)*pd) for y,ps,pd in held]
        cvrows.append({'alpha_static':a,'mean_event_auc':float(np.mean(auc)),'event_aucs':[float(x) for x in auc]})
    # Deterministic tie-break: prefer the more balanced mixture.
    best=max(cvrows,key=lambda r:(round(r['mean_event_auc'],12),-abs(r['alpha_static']-.5)))
    XtS,XtD,yt,pairs=test[0]['static'],test[0]['structured'],test[1],test[2]
    ps,hps=predict(events,XtS,'static',model,qk); pd,hpd=predict(events,XtD,'structured',model,qk)
    pe=best['alpha_static']*ps+(1-best['alpha_static'])*pd
    auc={'static':float(roc_auc_score(yt,ps)),'structured':float(roc_auc_score(yt,pd)),'ensemble':float(roc_auc_score(yt,pe))}
    return {'alpha_static':best['alpha_static'],'alpha_cv':cvrows,'auc':auc,
      'ensemble_minus_static_ci':base.bootdiff(yt,pe,ps,pairs,seed=701),
      'ensemble_minus_structured_ci':base.bootdiff(yt,pe,pd,pairs,seed=702),
      'static_hyperparams':hps,'structured_hyperparams':hpd}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--qsvm',required=True);p.add_argument('--out',default='isone_ensemble.json');p.add_argument('--seeds',default='300,301,302,303,304');a=p.parse_args()
    sys.path.insert(0,str(Path(a.qsvm).resolve().parent));import qsvm_kernel_v2 as qk
    d=Path(a.data);paths=[d/f'ISO-NE_case{i}.csv' for i in range(1,6)];common=sorted(set.intersection(*[set(base.usable_pmus(x)) for x in paths]));raw=[base.read_case(x,keep_indices=common) for x in paths]
    out={'design':{'experts':['static','structured'],'alpha_grid':ALPHAS,'selection':'nested leave-one-event-out; base models retuned inside every ensemble-weight fold','bootstrap':'source-window pairs'},'runs':[]}
    for seed in map(int,a.seeds.split(',')):
        ev=[ns.make_event(*r,seed+10*i,False) for i,r in enumerate(raw)]; run={'seed':seed,'folds':[]}
        for h in range(5):
            te=ns.make_event(*raw[h],seed+100+h,True); tr=[ev[i] for i in range(5) if i!=h]; fold={'heldout_case':h+1,'models':{}}
            for model in ['LR','RBF','QSVM']: fold['models'][model]=ensemble_fold(tr,te,model,qk)
            run['folds'].append(fold)
        out['runs'].append(run)
    Path(a.out).write_text(json.dumps(out,indent=2));print('Wrote',a.out)
if __name__=='__main__':main()
