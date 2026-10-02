"""Grid-extension check for baselines whose selected values hit a grid edge."""
import json, sys, numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import extra_models as em, synth_r4_prep as sp
seeds=[int(s) for s in sys.argv[1].split(',')]; OUT=sys.argv[2]
EXT={"Laplacian-SVM":{"C":[0.1,1,10,100],"gamma":[1/32,1/16,1/8,1/4,1/2,1,2]},
     "GBoost":{"learning_rate":[0.01,0.02,0.05,0.1],"max_depth":[1,2,3],"max_iter":[50,100,300]},
     "MLP":{"hidden_layer_sizes":[(16,),(32,),(64,)],"alpha":[1,3,10,30]}}
res=[]
import os
done=set(json.loads(l)['seed'] for l in open(OUT+'l')) if os.path.exists(OUT+'l') else set()
for seed in seeds:
    if seed in done: continue
    Z,y,Zt,yt=sp.prep(seed); cv=StratifiedKFold(3,shuffle=True,random_state=seed); r={'seed':seed,'auc':{},'sel':{}}
    for name,params in EXT.items():
        est,_,how=em.specs(seed)[name]; g=GridSearchCV(clone(est),params,cv=cv,scoring='roc_auc').fit(Z,y)
        r['auc'][name]=roc_auc_score(yt,em.score(g.best_estimator_,Zt,how)); r['sel'][name]={k:str(v) for k,v in g.best_params_.items()}
    res.append(r); open(OUT+'l','a').write(json.dumps(r)+'\n'); print(seed,{k:round(v,4) for k,v in r['auc'].items()},r['sel'],flush=True)
json.dump(res,open(OUT,'w'),indent=1)
