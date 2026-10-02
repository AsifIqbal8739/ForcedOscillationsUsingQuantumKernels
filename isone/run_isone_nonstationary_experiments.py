#!/usr/bin/env python3
"""ISO-NE v2: static vs multiscale vs structured-dynamic quantum kernels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,detrend
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import roc_auc_score
import run_isone_semisynthetic_validation as base

FS=30.; BAND=(.05,2.5); OUTER_SEC=60; STRIDE_SEC=20; NQ=8

def block_stats(VA):
    A=np.unwrap(np.deg2rad(VA),axis=0); sos=butter(4,BAND,fs=FS,btype='bandpass',output='sos')
    A=sosfiltfilt(sos,detrend(A,axis=0),axis=0); net=A.mean(1)
    Y=np.abs(np.fft.rfft(net)); fr=np.fft.rfftfreq(len(net),1/FS); ok=(fr>=BAND[0])&(fr<=BAND[1]); ids=np.flatnonzero(ok)
    peaks=ids[np.argsort(Y[ids])[-2:]][::-1]; vals=[]
    for k in peaks:
        z=np.fft.rfft(A,axis=0)[k]; ph=np.angle(z); mp=np.angle(np.mean(np.exp(1j*ph))); rel=np.angle(np.exp(1j*(ph-mp)))
        vals += [fr[k],1-np.abs(np.mean(np.exp(1j*ph))),np.quantile(np.abs(rel),.5),np.quantile(np.abs(rel),.9)]
    return np.asarray(vals) # [f,disp,q50,q90] x two peaks

def spectral_entropy(x):
    p=np.abs(np.fft.rfft(detrend(x)))**2+1e-12; p=p/p.sum(); return float(-(p*np.log(p)).sum()/np.log(len(p)))

def reps(F,VM,VA):
    n=len(VA); static=block_stats(VA[-int(20*FS):])
    multi=np.r_[block_stats(VA[-int(20*FS):]),block_stats(VA[-int(40*FS):]),block_stats(VA)]
    chunks=np.array_split(VA,4); B=np.vstack([block_stats(c) for c in chunks]);
    f=B[:,0]; disp=B[:,1]; q90=B[:,3]
    # Eight interpretable dynamic inputs; no PCA is used for this branch.
    structured=np.array([f.mean(),f.std(),np.polyfit(np.arange(4),f,1)[0],disp.mean(),
      np.polyfit(np.arange(4),disp,1)[0],q90.mean(),np.polyfit(np.arange(4),q90,1)[0],spectral_entropy(VA.mean(1))])
    return {'static':static,'multiscale':multi,'structured':structured}

def make_event(F,VM,VA,seed,shifted=False):
    rng=np.random.default_rng(seed); L=int(OUTER_SEC*FS); st=int(STRIDE_SEC*FS); out={k:[] for k in ['static','multiscale','structured']}; y=[]; pair=[]
    n=VA.shape[1]; width=max(1,n//2); start=n-width if shifted else 0; attacked=np.arange(start,start+width)
    for i in range(0,len(VA)-L+1,st):
        fw,vw,aw=F[i:i+L],VM[i:i+L],VA[i:i+L]; rg=reps(fw,vw,aw)
        for k in out: out[k].append(rg[k])
        pid=len(pair)//2; y.append(1);pair.append(pid)
        off=np.zeros(n); off[attacked]=rng.uniform(.008,.020)+np.linspace(-1,1,width)*rng.uniform(.008,.020)+rng.normal(0,.002,width)
        rg=reps(base.shift_columns(fw,off,attacked),base.shift_columns(vw,off,attacked),base.shift_columns(aw,off,attacked))
        for k in out: out[k].append(rg[k])
        y.append(0);pair.append(pid)
    return {k:np.asarray(v) for k,v in out.items()},np.asarray(y),np.asarray(pair)

def fit_transform(X,nq):
    s1=StandardScaler().fit(X); z=s1.transform(X)
    pc=None if X.shape[1]==nq else PCA(n_components=nq).fit(z); z=z if pc is None else pc.transform(z)
    s2=StandardScaler().fit(z); return (s1,pc,s2),s2.transform(z)
def apply_transform(X,t):
    s1,pc,s2=t; z=s1.transform(X); return s2.transform(z if pc is None else pc.transform(z))
def edges_for(kind,n):
    if kind=='structured': return [(0,1),(1,2),(3,4),(5,6),(0,3),(3,5),(2,7),(4,7),(6,7)]
    return [(i,(i+1)%n) for i in range(n)]
def qselect(X,y,g,kind,qk):
    nq=min(NQ,X.shape[1]); ed=edges_for(kind,nq); rows=[]; logo=LeaveOneGroupOut()
    for beta in [.5,1]:
     for bp in [.5,1]:
      for C in [.1,1,10]:
       aa=[]
       for tr,va in logo.split(X,y,g):
        tf,Z=fit_transform(X[tr],nq); Zv=apply_transform(X[va],tf)
        _,p,_,_,_,_,_=qk.fit_and_eval(Z,y[tr],Zv,beta,bp,ed,C,2,None); aa.append(roc_auc_score(y[va],p))
       rows.append({'beta':beta,'beta_prime':bp,'C':C,'mean_auc':float(np.mean(aa))})
    return max(rows,key=lambda r:r['mean_auc']),ed
def eval_fold(events,test,kind,qk):
    X=np.vstack([e[0][kind] for e in events]); y=np.hstack([e[1] for e in events]); g=np.hstack([np.full(len(e[1]),i) for i,e in enumerate(events)])
    Xt,yt,pairs=test[0][kind],test[1],test[2]; nq=min(NQ,X.shape[1]); splits=list(LeaveOneGroupOut().split(X,y,g))
    prep=[('s',StandardScaler())]
    if X.shape[1]>nq: prep.append(('pca',PCA(n_components=nq)))
    lr=GridSearchCV(Pipeline(prep+[('m',LogisticRegression(max_iter=3000,class_weight='balanced'))]),{'m__C':[.1,1,10]},cv=splits,scoring='roc_auc').fit(X,y)
    rb=GridSearchCV(Pipeline(prep+[('m',SVC(kernel='rbf',class_weight='balanced'))]),{'m__C':[.1,1,10],'m__gamma':['scale',.1,1]},cv=splits,scoring='roc_auc').fit(X,y)
    best,ed=qselect(X,y,g,kind,qk); tf,Z=fit_transform(X,nq); Zt=apply_transform(Xt,tf)
    _,pq,_,align,_,_,_=qk.fit_and_eval(Z,y,Zt,best['beta'],best['beta_prime'],ed,best['C'],2,None)
    pl=lr.predict_proba(Xt)[:,1]; pr=base.sigmoid(rb.decision_function(Xt))
    return {'n_train':len(y),'n_test':len(yt),'auc':{'LR':float(roc_auc_score(yt,pl)),'RBF':float(roc_auc_score(yt,pr)),'QSVM':float(roc_auc_score(yt,pq))},
      'qsvm_minus_lr_ci':base.bootdiff(yt,pq,pl,pairs),'qsvm_minus_rbf_ci':base.bootdiff(yt,pq,pr,pairs),'qsvm_hyperparams':best,'edges':ed,'alignment':float(align)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--qsvm',required=True);p.add_argument('--out',default='isone_nonstationary.json');p.add_argument('--seeds',default='200,201,202,203,204');a=p.parse_args()
    sys.path.insert(0,str(Path(a.qsvm).resolve().parent));import qsvm_kernel_v2 as qk
    d=Path(a.data);paths=[d/f'ISO-NE_case{i}.csv' for i in range(1,6)];common=sorted(set.intersection(*[set(base.usable_pmus(x)) for x in paths]));raw=[base.read_case(x,keep_indices=common) for x in paths]
    out={'design':{'outer_window_sec':OUTER_SEC,'stride_sec':STRIDE_SEC,'representations':['static','multiscale','structured'],'split':'nested leave-one-event-out'},'runs':[]}
    for seed in map(int,a.seeds.split(',')):
        train=[make_event(*r,seed+10*i,False) for i,r in enumerate(raw)]; rr={'seed':seed,'folds':[]}
        for h in range(5):
            test=make_event(*raw[h],seed+100+h,True); z={'heldout_case':h+1,'representations':{}}
            for kind in ['static','multiscale','structured']: z['representations'][kind]=eval_fold([train[i] for i in range(5) if i!=h],test,kind,qk)
            rr['folds'].append(z)
        out['runs'].append(rr)
    Path(a.out).write_text(json.dumps(out,indent=2));print('Wrote',a.out)
if __name__=='__main__':main()
