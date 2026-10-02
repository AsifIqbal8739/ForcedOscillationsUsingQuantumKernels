#!/usr/bin/env python3
"""Projected and multiple-small-quantum-kernel ISO-NE experiment."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import run_isone_semisynthetic_validation as base
import run_isone_nonstationary_experiments as ns

def observables(X,beta,edges,qk):
    S=qk.states_matrix(X,beta,1.0,edges,2); n=X.shape[1]; idx=np.arange(1<<n); O=[]
    for a in S:
        p=np.abs(a)**2; row=[]
        for i in range(n):
            z=1-2*((idx>>i)&1); row.append(float(p@z)); row.append(float(np.real(np.vdot(a,a[idx^(1<<i)]))))
        for i,j in edges:
            zz=(1-2*((idx>>i)&1))*(1-2*((idx>>j)&1)); row.append(float(p@zz))
        O.append(row)
    return np.asarray(O)
def rbf(A,B,gamma):
    d=np.maximum((A*A).sum(1)[:,None]+(B*B).sum(1)[None,:]-2*A@B.T,0); return np.exp(-gamma*d)
def center(Ktr,Kte=None):
    mu=Ktr.mean(0); bar=Ktr.mean(); H=Ktr-mu[None,:]-mu[:,None]+bar
    if Kte is None:return H
    return H,Kte-Kte.mean(1)[:,None]-mu[None,:]+bar
def score_kernel(Ktr,y,Kte,C):
    clf=SVC(kernel='precomputed',C=C,class_weight='balanced').fit(Ktr,y); return base.sigmoid(clf.decision_function(Kte))

def kernels(Xtr,Xte,design,pars,qk):
    if design=='fidelity':
        edges=ns.edges_for('structured',8); St=qk.states_matrix(Xtr,pars['beta'],pars['bp'],edges,2); Se=qk.states_matrix(Xte,pars['beta'],pars['bp'],edges,2)
        A=qk.kernel_matrix_from_states(St,St); B=qk.kernel_matrix_from_states(Se,St); _,Bt,_,_,_=qk.normalize_center_kernel(A,B,np.ones(len(Xte))); At=qk.normalize_center_kernel(A)[0]; return At,Bt
    groups=[np.array([0,1,2,7]),np.array([3,4,5,6])]
    Ks=[]
    for g in groups:
        ed=[(i,(i+1)%4) for i in range(4)]; Ot=observables(Xtr[:,g],pars['beta'],ed,qk); Oe=observables(Xte[:,g],pars['beta'],ed,qk)
        A=rbf(Ot,Ot,pars['gamma']);B=rbf(Oe,Ot,pars['gamma']);Ks.append(center(A,B))
    if design=='projected':
        A=.5*Ks[0][0]+.5*Ks[1][0];B=.5*Ks[0][1]+.5*Ks[1][1]
    else:
        w=pars['weight'];A=w*Ks[0][0]+(1-w)*Ks[1][0];B=w*Ks[0][1]+(1-w)*Ks[1][1]
    return A,B

def candidates(design):
    if design=='fidelity': return [{'beta':b,'bp':bp,'C':c} for b in [.5,1] for bp in [.5,1] for c in [.1,1,10]]
    weights=[.5] if design=='projected' else [0,.25,.5,.75,1]
    return [{'beta':b,'gamma':g,'C':c,'weight':w} for b in [.5,1] for g in [.1,1] for c in [.1,1,10] for w in weights]
def robust_select(X,y,groups,design,qk):
    rows=[];logo=LeaveOneGroupOut()
    for p in candidates(design):
        aa=[]
        for tr,va in logo.split(X,y,groups):
            sc=StandardScaler().fit(X[tr]);Z=sc.transform(X[tr]);Zv=sc.transform(X[va]);A,B=kernels(Z,Zv,design,p,qk);aa.append(roc_auc_score(y[va],score_kernel(A,y[tr],B,p['C'])))
        mean=float(np.mean(aa));worst=float(np.min(aa));rows.append({**p,'mean_event_auc':mean,'worst_event_auc':worst,'objective':.5*(mean+worst)})
    return max(rows,key=lambda r:r['objective']),rows

def augment_event(raw,seed,naug=3,shifted=False):
    copies=[ns.make_event(*raw,seed+100*k,shifted) for k in range(naug)]; out={k:[] for k in copies[0][0]};y=[];pair=[]
    # One balanced genuine/spoof pair per augmentation; all stay in one event fold.
    for k,c in enumerate(copies):
        for name in out:out[name].append(c[0][name])
        y.append(c[1]);pair.append(c[2]+k*(c[2].max()+1))
    return {k:np.vstack(v) for k,v in out.items()},np.hstack(y),np.hstack(pair)
def eval_design(events,test,design,qk):
    X=np.vstack([e[0]['structured'] for e in events]);y=np.hstack([e[1] for e in events]);groups=np.hstack([np.full(len(e[1]),i) for i,e in enumerate(events)])
    Xt,yt,pairs=test[0]['structured'],test[1],test[2];best,grid=robust_select(X,y,groups,design,qk);sc=StandardScaler().fit(X);Z=sc.transform(X);Zt=sc.transform(Xt);A,B=kernels(Z,Zt,design,best,qk);p=score_kernel(A,y,B,best['C'])
    return {'auc':float(roc_auc_score(yt,p)),'selected':best,'q10_grid_objective':sorted(grid,key=lambda r:r['objective'],reverse=True)[:10],'n_train':len(y),'n_test':len(yt)} ,p
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--qsvm',required=True);ap.add_argument('--out',default='isone_projected_quantum.json');ap.add_argument('--seeds',default='400,401,402,403,404');ap.add_argument('--augment',type=int,default=3);a=ap.parse_args()
    sys.path.insert(0,str(Path(a.qsvm).resolve().parent));import qsvm_kernel_v2 as qk
    d=Path(a.data);paths=[d/f'ISO-NE_case{i}.csv' for i in range(1,6)];common=sorted(set.intersection(*[set(base.usable_pmus(p)) for p in paths]));raw=[base.read_case(p,keep_indices=common) for p in paths]
    out={'design':{'quantum_kernels':['fidelity','projected','multiple_projected'],'augmentation_per_window':a.augment,'selection_objective':'0.5*mean event AUC + 0.5*worst event AUC','features':'structured dynamic'},'runs':[]}
    for seed in map(int,a.seeds.split(',')):
        ev=[augment_event(r,seed+10*i,a.augment,False) for i,r in enumerate(raw)];run={'seed':seed,'folds':[]}
        for h in range(5):
            te=augment_event(raw[h],seed+1000+h,1,True);tr=[ev[i] for i in range(5) if i!=h];fold={'heldout_case':h+1,'kernels':{}};pred={}
            for design in ['fidelity','projected','multiple_projected']:
                fold['kernels'][design],pred[design]=eval_design(tr,te,design,qk)
            yt,pairs=te[1],te[2];
            for name in ['projected','multiple_projected']:
                fold['kernels'][name]['minus_fidelity_ci']=base.bootdiff(yt,pred[name],pred['fidelity'],pairs,seed=900+h)
            run['folds'].append(fold)
        out['runs'].append(run)
    Path(a.out).write_text(json.dumps(out,indent=2));print('Wrote',a.out)
if __name__=='__main__':main()
