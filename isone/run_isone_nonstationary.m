function status=run_isone_nonstationary(dataDir,varargin)
p=inputParser;addParameter(p,'Seeds','200,201,202,203,204');addParameter(p,'Output','isone_nonstationary.json');addParameter(p,'Python','');parse(p,varargin{:});o=p.Results;
root=fileparts(mfilename('fullpath'));driver=fullfile(root,'run_isone_nonstationary_experiments.py');q=fullfile(root,'qsvm_kernel_v2.py');
if strlength(string(o.Python))>0,PY=string(o.Python);else,e=string(pyenv().Executable);f="/opt/anaconda3/envs/qiskit_matlab/bin/python3";if strlength(e)>0&&isfile(e),PY=e;else,PY=f;end,end
cmd=sprintf('"%s" "%s" --data "%s" --qsvm "%s" --out "%s" --seeds "%s"',PY,driver,dataDir,q,fullfile(root,o.Output),o.Seeds);
[status,msg]=system(cmd);fprintf('%s\n',msg);assert(status==0,'Nonstationary experiment failed: %s',msg);
end
