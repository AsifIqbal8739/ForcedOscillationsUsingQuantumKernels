import json, sys, numpy as np
from sklearn.metrics import roc_auc_score
def load(p): return json.load(open(p))['runs']
def fold_auc_diff(f, a, b, ix):
    y=np.asarray(f['y'])[ix]
    if len(np.unique(y))<2: return None
    return roc_auc_score(y,np.asarray(f['pred'][a])[ix])-roc_auc_score(y,np.asarray(f['pred'][b])[ix])
def agg(runs, a, b, B=10000, seed=7):
    r=np.random.default_rng(seed); S=len(runs)
    units=[[ [np.flatnonzero(np.asarray(f['pairs'])==u) for u in np.unique(f['pairs'])] for f in run['folds']] for run in runs]
    point=np.mean([[fold_auc_diff(f,a,b,np.arange(len(f['y']))) for f in run['folds']] for run in runs])
    boots=[]; boots_seed=[]
    per_seed=np.array([np.mean([fold_auc_diff(f,a,b,np.arange(len(f['y']))) for f in run['folds']]) for run in runs])
    for _ in range(B):
        ss=r.integers(0,S,S); vals=[]
        for s in ss:
            ev=[]
            for fi,f in enumerate(runs[s]['folds']):
                U=units[s][fi]; pick=r.integers(0,len(U),len(U)); ix=np.concatenate([U[k] for k in pick])
                d=fold_auc_diff(f,a,b,ix)
                if d is not None: ev.append(d)
            vals.append(np.mean(ev))
        boots.append(np.mean(vals)); boots_seed.append(per_seed[ss].mean())
    boots=np.array(boots)
    p=2*min((boots<=0).mean(),(boots>=0).mean())
    per_event=np.mean([[fold_auc_diff(f,a,b,np.arange(len(f['y']))) for f in run['folds']] for run in runs],axis=0)
    return {'point':float(point),'ci_hier':np.quantile(boots,[.025,.975]).round(4).tolist(),'p_boot':float(max(p,1/B)),
            'ci_seed_only':np.quantile(boots_seed,[.025,.975]).round(4).tolist(),'per_event':per_event.round(4).tolist(),
            'events_positive':int((per_event>0).sum())}
if __name__=='__main__':
    runs=load(sys.argv[1])
    for a,b in [('QSVM','RBF'),('QSVM','LR'),('RBF','LR')]:
        print(a,'-',b, agg(runs,a,b))
