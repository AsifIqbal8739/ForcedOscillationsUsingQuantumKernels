#!/usr/bin/env python3
"""Final matched LR/RBF/fidelity-QSVM comparison on ISO-NE structured features."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import run_isone_semisynthetic_validation as base
import run_isone_nonstationary_experiments as ns
import run_isone_projected_quantum as pq

def stack(events):
    X=np.vstack([e[0]['structured'] for e in events]);y=np.hstack([e[1] for e in events]);g=np.hstack([np.full(len(e[1]),i) for i,e in enumerate(events)]);return X,y,g
def objective(aucs):return .5*(float(np.mean(aucs))+float(np.min(aucs)))
def select_classical(X,y,g,model):
    cand=[]; logo=LeaveOneGroupOut()
    grid=[{'C':c} for c in [.1,1,10]] if model=='LR' else [{'C':c,'gamma':ga} for c in [.1,1,10] for ga in ['scale',.1,1]]
    for p in grid:
        aa=[]
        for tr,va in logo.split(X,y,g):
            sc=StandardScaler().fit(X[tr]);Z=sc.transform(X[tr]);Zv=sc.transform(X[va])
            clf=LogisticRegression(C=p['C'],max_iter=3000,class_weight='balanced') if model=='LR' else SVC(C=p['C'],gamma=p['gamma'],kernel='rbf',class_weight='balanced')
            clf.fit(Z,y[tr]);score=clf.predict_proba(Zv)[:,1] if model=='LR' else base.sigmoid(clf.decision_function(Zv));aa.append(roc_auc_score(y[va],score))
        cand.append({**p,'mean_event_auc':float(np.mean(aa)),'worst_event_auc':float(np.min(aa)),'objective':objective(aa)})
    return max(cand,key=lambda r:r['objective']),cand
def fit_classical(X,y,Xt,model,p):
    sc=StandardScaler().fit(X);Z=sc.transform(X);Zt=sc.transform(Xt)
    clf=LogisticRegression(C=p['C'],max_iter=3000,class_weight='balanced') if model=='LR' else SVC(C=p['C'],gamma=p['gamma'],kernel='rbf',class_weight='balanced')
    clf.fit(Z,y);return clf.predict_proba(Zt)[:,1] if model=='LR' else base.sigmoid(clf.decision_function(Zt))
def eval_fold(events,test,qk):
    X,y,g=stack(events);Xt,yt,pairs=test[0]['structured'],test[1],test[2];out={'models':{}};pred={}
    for model in ['LR','RBF']:
        best,grid=select_classical(X,y,g,model);pred[model]=fit_classical(X,y,Xt,model,best);out['models'][model]={'auc':float(roc_auc_score(yt,pred[model])),'selected':best,'grid':grid}
    best,grid=pq.robust_select(X,y,g,'fidelity',qk);sc=StandardScaler().fit(X);Z=sc.transform(X);Zt=sc.transform(Xt);A,B=pq.kernels(Z,Zt,'fidelity',best,qk);pred['QSVM']=pq.score_kernel(A,y,B,best['C'])
    out['models']['QSVM']={'auc':float(roc_auc_score(yt,pred['QSVM'])),'selected':best,'q10_grid_objective':sorted(grid,key=lambda r:r['objective'],reverse=True)[:10]}
    out['qsvm_minus_lr_ci']=base.bootdiff(yt,pred['QSVM'],pred['LR'],pairs,seed=811)
    out['qsvm_minus_rbf_ci']=base.bootdiff(yt,pred['QSVM'],pred['RBF'],pairs,seed=812)
    out['strict_qsvm_win']=bool(out['models']['QSVM']['auc']>max(out['models']['LR']['auc'],out['models']['RBF']['auc']) and out['qsvm_minus_lr_ci'][0]>0 and out['qsvm_minus_rbf_ci'][0]>0)
    out['n_train']=len(y);out['n_test']=len(yt);return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--qsvm',required=True);ap.add_argument('--out',default='isone_final_matched.json');ap.add_argument('--seeds',default='400,401,402,403,404');ap.add_argument('--augment',type=int,default=3);a=ap.parse_args()
    sys.path.insert(0,str(Path(a.qsvm).resolve().parent));import qsvm_kernel_v2 as qk
    d=Path(a.data);paths=[d/f'ISO-NE_case{i}.csv' for i in range(1,6)];common=sorted(set.intersection(*[set(base.usable_pmus(p)) for p in paths]));raw=[base.read_case(p,keep_indices=common) for p in paths]
    out={'design':{'models':['LR','RBF','fidelity_QSVM'],'features':'structured dynamic','augmentation_per_window':a.augment,'seeds':a.seeds,'selection_objective':'0.5*mean event AUC + 0.5*worst event AUC','splits':'nested complete-event','bootstrap':'source-window pairs'},'runs':[]}
    for seed in map(int,a.seeds.split(',')):
        ev=[pq.augment_event(r,seed+10*i,a.augment,False) for i,r in enumerate(raw)];run={'seed':seed,'folds':[]}
        for h in range(5):
            te=pq.augment_event(raw[h],seed+1000+h,1,True);run['folds'].append({'heldout_case':h+1,**eval_fold([ev[i] for i in range(5) if i!=h],te,qk)})
        out['runs'].append(run)
    Path(a.out).write_text(json.dumps(out,indent=2));print('Wrote',a.out)
if __name__=='__main__':main()
