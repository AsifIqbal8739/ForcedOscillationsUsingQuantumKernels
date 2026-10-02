import json, agg_ci, sys
runs=agg_ci.load(sys.argv[1]); res={}
for a,b in [tuple(x.split('-',1)) for x in sys.argv[3].split(',')]:
    res[f'{a}-{b}']=agg_ci.agg(runs,a,b,B=2000); print(a,b,res[f'{a}-{b}'],flush=True)
json.dump(res,open(sys.argv[2],'w'),indent=1)
