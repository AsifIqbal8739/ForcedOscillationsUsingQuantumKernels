#!/usr/bin/env python3
"""Event-held-out ISO-NE validation with semi-synthetic PMU timing attacks.

Genuine windows come directly from ISO-NE records. Cyber counterparts are
created by fractionally time-shifting a contiguous PMU footprint. Splits are
by complete event, so no source window appears in both train and test.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

FS=30.0; WIN_SEC=20; STRIDE_SEC=10; BAND=(0.1,2.5); NQ=8

def read_case(path, max_pmus=None, keep_indices=None):
    # Four nonnumeric/header rows precede the samples. Columns repeat F,VM,VA,IM,IA.
    a=pd.read_csv(path,header=None,skiprows=4,low_memory=False).apply(pd.to_numeric,errors='coerce').to_numpy()
    a=a[np.isfinite(a[:,0])]; n=(a.shape[1]-1)//5
    if max_pmus: n=min(n,max_pmus)
    F=np.column_stack([a[:,1+5*i] for i in range(n)])
    VM=np.column_stack([a[:,2+5*i] for i in range(n)])
    VA=np.column_stack([a[:,3+5*i] for i in range(n)])
    if keep_indices is not None:
        F,VM,VA=F[:,keep_indices],VM[:,keep_indices],VA[:,keep_indices]
    for x in (F,VM,VA):
        for j in range(x.shape[1]):
            bad=~np.isfinite(x[:,j]); good=~bad
            if not good.any():
                raise ValueError(f'{path}: selected PMU column {j} has no numeric samples')
            if bad.any(): x[bad,j]=np.interp(np.flatnonzero(bad),np.flatnonzero(good),x[good,j])
    return F,VM,VA

def usable_pmus(path):
    """Return PMU-group indices with adequate F, VM and VA observations."""
    a=pd.read_csv(path,header=None,skiprows=4,low_memory=False).apply(pd.to_numeric,errors='coerce').to_numpy()
    a=a[np.isfinite(a[:,0])]; n=(a.shape[1]-1)//5; keep=[]
    for i in range(n):
        cols=a[:,[1+5*i,2+5*i,3+5*i]]
        if np.all(np.mean(np.isfinite(cols),axis=0)>=0.90): keep.append(i)
    return keep

def shift_columns(x, offsets_sec, attacked):
    """Fractional trajectory shift; edge values are held, avoiding wraparound."""
    y=x.copy(); t=np.arange(len(x))/FS
    for j in attacked:
        y[:,j]=np.interp(t+offsets_sec[j],t,x[:,j],left=x[0,j],right=x[-1,j])
    return y

def phase_features(F,VM,VA):
    sos=butter(4,BAND,fs=FS,btype='bandpass',output='sos')
    # Angle trajectories carry the inter-PMU phase geometry. Unwrap before filtering.
    A=np.unwrap(np.deg2rad(VA),axis=0); Ab=sosfiltfilt(sos,detrend(A,axis=0),axis=0)
    network=np.mean(Ab,axis=1); Y=np.abs(np.fft.rfft(network)); fr=np.fft.rfftfreq(len(network),1/FS)
    m=(fr>=BAND[0])&(fr<=BAND[1]); k=np.flatnonzero(m)[np.argmax(Y[m])]
    z=np.fft.rfft(Ab,axis=0)[k]; ph=np.angle(z); meanph=np.angle(np.mean(np.exp(1j*ph)))
    rel=np.angle(np.exp(1j*(ph-meanph)))
    # Position-indexed circular block plus invariant phase summaries.
    return np.r_[np.cos(rel),np.sin(rel),1-np.abs(np.mean(np.exp(1j*ph))),np.std(np.sort(rel)),np.max(np.abs(rel)),fr[k]]

def make_event(F,VM,VA,seed,shifted_footprint=False):
    rng=np.random.default_rng(seed); L=int(WIN_SEC*FS); st=int(STRIDE_SEC*FS); X=[]; y=[]; pair=[]
    n=F.shape[1]; width=max(1,n//2); start=(n-width if shifted_footprint else 0); attacked=np.arange(start,start+width)
    for i in range(0,len(F)-L+1,st):
        fw,vw,aw=F[i:i+L],VM[i:i+L],VA[i:i+L]
        pid=len(pair)//2; X.append(phase_features(fw,vw,aw)); y.append(1); pair.append(pid) # genuine
        # 8--40 ms PMU-specific timing errors with a footprint gradient.
        base=rng.uniform(.008,.020); grad=np.linspace(-1,1,width)*rng.uniform(.008,.020)
        offs=np.zeros(n); offs[attacked]=base+grad+rng.normal(0,.002,width)
        X.append(phase_features(shift_columns(fw,offs,attacked),shift_columns(vw,offs,attacked),shift_columns(aw,offs,attacked))); y.append(0); pair.append(pid)
    return np.asarray(X),np.asarray(y),np.asarray(pair)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))
def bootdiff(y,a,b,pairs,B=2000,seed=9):
    """Cluster bootstrap: resample source-window pairs, retaining both rows."""
    r=np.random.default_rng(seed); z=[]; units=np.unique(pairs); n=len(units)
    for _ in range(B):
        chosen=r.choice(units,size=n,replace=True)
        ix=np.concatenate([np.flatnonzero(pairs==u) for u in chosen])
        if len(np.unique(y[ix]))==2: z.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
    return [float(x) for x in np.quantile(z,[.025,.975])]

def transform_fit(X,nq):
    s1=StandardScaler().fit(X); z=s1.transform(X); pc=PCA(n_components=nq).fit(z); z=pc.transform(z); s2=StandardScaler().fit(z)
    return s1,pc,s2,s2.transform(z)

def transform_apply(X,s1,pc,s2): return s2.transform(pc.transform(s1.transform(X)))

def q_group_select(X,y,groups,edges,qk,seed,nq):
    """Quantum selection with event folds and fold-local preprocessing."""
    logo=LeaveOneGroupOut(); rows=[]
    for beta in [.5,1,2]:
      for bp in [.5,1,2]:
       for C in [.1,1,10]:
        auc=[]
        for tr,va in logo.split(X,y,groups):
            s1,pc,s2,Ztr=transform_fit(X[tr],nq); Zva=transform_apply(X[va],s1,pc,s2)
            _,p,_,_,_,_,_=qk.fit_and_eval(Ztr,y[tr],Zva,beta,bp,edges,C,2,None)
            auc.append(roc_auc_score(y[va],p))
        rows.append({'beta':beta,'beta_prime':bp,'C':C,'mean_auc':float(np.mean(auc))})
    return max(rows,key=lambda z:z['mean_auc']),rows

def evaluate_fold(train_events,test_event,qk,seed,max_train):
    X=np.vstack([z[0] for z in train_events]); y=np.hstack([z[1] for z in train_events]);
    groups=np.hstack([np.full(len(z[1]),i) for i,z in enumerate(train_events)]); Xt,yt,test_pairs=test_event
    if len(y)>max_train:
        # Subsample complete genuine/spoof pairs within every event.
        r=np.random.default_rng(seed); chosen=[]; per=max(1,(max_train//2)//len(train_events))
        off=0
        for gi,z in enumerate(train_events):
            pu=np.unique(z[2]); take=r.choice(pu,min(per,len(pu)),False)
            chosen.extend((off+np.flatnonzero(np.isin(z[2],take))).tolist()); off+=len(z[1])
        ix=np.asarray(chosen); r.shuffle(ix); X,y,groups=X[ix],y[ix],groups[ix]
    nq=min(NQ,X.shape[1],len(X)-2); sc,pc,sc2,Z=transform_fit(X,nq); Zt=transform_apply(Xt,sc,pc,sc2)
    cv=list(LeaveOneGroupOut().split(Z,y,groups))
    lrpipe=Pipeline([('s1',StandardScaler()),('pca',PCA(n_components=nq)),('s2',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))])
    rbpipe=Pipeline([('s1',StandardScaler()),('pca',PCA(n_components=nq)),('s2',StandardScaler()),('model',SVC(kernel='rbf',class_weight='balanced'))])
    lr=GridSearchCV(lrpipe,{'model__C':[.1,1,10]},cv=cv,scoring='roc_auc').fit(X,y)
    rb=GridSearchCV(rbpipe,{'model__C':[.1,1,10],'model__gamma':['scale',.1,1]},cv=cv,scoring='roc_auc').fit(X,y)
    # PCA components have no descriptor-specific physical adjacency: use a generic ring.
    edges=[(i,(i+1)%nq) for i in range(nq)]
    best,_=q_group_select(X,y,groups,edges,qk,seed,nq)
    _,pq,_,align,tt,te,_=qk.fit_and_eval(Z,y,Zt,best['beta'],best['beta_prime'],edges,best['C'],2,None)
    pl=lr.predict_proba(Xt)[:,1]; pr=sigmoid(rb.decision_function(Xt))
    return {'n_train':len(y),'n_test':len(yt),'n_qubits':nq,'pca_variance':float(pc.explained_variance_ratio_.sum()),
      'auc':{'LR':float(roc_auc_score(yt,pl)),'RBF':float(roc_auc_score(yt,pr)),'QSVM':float(roc_auc_score(yt,pq))},
      'qsvm_minus_lr_ci':bootdiff(yt,pq,pl,test_pairs,seed=seed+1),'qsvm_minus_rbf_ci':bootdiff(yt,pq,pr,test_pairs,seed=seed+2),
      'qsvm_hyperparams':best,'kernel_alignment':float(align),'timing':{'train':tt,'test':te}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--qsvm',required=True); p.add_argument('--out',default='isone_validation.json'); p.add_argument('--seeds',default='100,101,102,103,104'); p.add_argument('--max-train',type=int,default=400); a=p.parse_args()
    sys.path.insert(0,str(Path(a.qsvm).resolve().parent)); import qsvm_kernel_v2 as qk
    d=Path(a.data)
    paths=[d/f'ISO-NE_case{i}.csv' for i in range(1,6)]
    common=sorted(set.intersection(*[set(usable_pmus(p)) for p in paths]))
    if len(common)<8: raise ValueError(f'Only {len(common)} PMU groups are usable across cases 1-5; need at least 8. Indices={common}')
    raw=[read_case(p,keep_indices=common) for p in paths]
    keep6=usable_pmus(d/'ISO-NE_case6.csv')
    if len(keep6)<4: raise ValueError(f'Case 6 has only {len(keep6)} usable PMU groups')
    raw6=read_case(d/'ISO-NE_case6.csv',keep_indices=keep6[:4])
    out={'design':{'primary':'leave-one-event-out cases 1-5, shifted spoof footprint at test','secondary':'train cases 1-5 first four PMUs, test case 6','label_1':'genuine','label_0':'semi-synthetic timing attack','common_pmu_indices_zero_based':common,'inner_cv':'leave-one-training-event-out; genuine/spoof pairs kept together','confidence_intervals':'cluster bootstrap of source-window pairs'},'runs':[]}
    for seed in [int(x) for x in a.seeds.split(',')]:
        # Train footprints occupy first half; held-out event uses the other half.
        tr_events=[make_event(*r,seed+10*i,False) for i,r in enumerate(raw)]
        folds=[]
        for held in range(5):
            test=make_event(*raw[held],seed+100+held,True)
            folds.append({'heldout_case':held+1,**evaluate_fold([tr_events[i] for i in range(5) if i!=held],test,qk,seed+held,a.max_train)})
        # Case 6 reduced-observability transfer, trained on matching first four channels.
        tr4=[make_event(r[0][:,:4],r[1][:,:4],r[2][:,:4],seed+200+i,False) for i,r in enumerate(raw)]
        te6=make_event(*raw6,seed+300,True); reduced=evaluate_fold(tr4,te6,qk,seed+99,a.max_train)
        out['runs'].append({'seed':seed,'primary_folds':folds,'case6_reduced_observability':reduced})
    Path(a.out).write_text(json.dumps(out,indent=2)); print(f'Wrote {a.out}')
if __name__=='__main__': main()
