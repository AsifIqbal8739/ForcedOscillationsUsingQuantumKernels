function status=run_isone_validation(dataDir,varargin)
% MATLAB wrapper for event-held-out ISO-NE semi-synthetic validation.
p=inputParser; addParameter(p,'Seeds','100,101,102,103,104'); addParameter(p,'MaxTrain',400); addParameter(p,'Output','isone_validation.json'); addParameter(p,'Python',''); parse(p,varargin{:}); o=p.Results;
root=fileparts(mfilename('fullpath')); driver=fullfile(root,'run_isone_semisynthetic_validation.py'); qsvm=fullfile(root,'qsvm_kernel_v2.py');
assert(isfile(driver),'Missing %s',driver); assert(isfile(qsvm),'Missing %s',qsvm);
for i=1:6, assert(isfile(fullfile(dataDir,sprintf('ISO-NE_case%d.csv',i))),'Missing ISO-NE case %d',i); end
if strlength(string(o.Python))>0, PY=string(o.Python); else, e=string(pyenv().Executable); f="/opt/anaconda3/envs/qiskit_matlab/bin/python3"; if strlength(e)>0&&isfile(e),PY=e;elseif isfile(f),PY=f;else,error('Pass a valid Python path.');end,end
out=fullfile(root,o.Output);
cmd=sprintf('"%s" "%s" --data "%s" --qsvm "%s" --out "%s" --seeds "%s" --max-train %d',PY,driver,dataDir,qsvm,out,o.Seeds,o.MaxTrain);
fprintf('Running ISO-NE validation:\n%s\n',cmd); [status,msg]=system(cmd); fprintf('%s\n',msg); assert(status==0,'ISO-NE validation failed: %s',msg);
fprintf('Saved %s\n',out);
end
