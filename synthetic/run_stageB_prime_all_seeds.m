function Results = run_stageB_prime_all_seeds(varargin)
% Automate Stage B' for multiple independent seeds and aggregate results.
%
% Example:
%   Results = run_stageB_prime_all_seeds;
%   Results = run_stageB_prime_all_seeds('Seeds',42:46,'MaxTrain',400);
%
% Required in the same directory:
%   run_stageB_prime_export.m, run_stageB_prime.py, qsvm_kernel_v2.py

p=inputParser;
addParameter(p,'Seeds',42:46,@(x)isnumeric(x)&&isvector(x));
addParameter(p,'MaxTrain',400,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'OutputDir','stageB_prime_multiseed',@(x)ischar(x)||isstring(x));
addParameter(p,'Python','',@(x)ischar(x)||isstring(x));
addParameter(p,'Force',false,@(x)islogical(x)&&isscalar(x));
parse(p,varargin{:}); opt=p.Results;

root=fileparts(mfilename('fullpath'));
pyDriver=fullfile(root,'run_stageB_prime.py');
pyQsvm=fullfile(root,'qsvm_kernel_v2.py');
assert(isfile(pyDriver),'Missing %s',pyDriver);
assert(isfile(pyQsvm),'Missing %s',pyQsvm);

if strlength(string(opt.Python))>0
    PY=string(opt.Python);
else
    pe=pyenv;
    % system() does not require MATLAB's in-process Python interface to be
    % loaded. Use its configured executable if present, then the established
    % qiskit_matlab environment used by this project.
    configured=string(pe.Executable);
    fallback="/opt/anaconda3/envs/qiskit_matlab/bin/python3";
    if strlength(configured)>0 && isfile(configured)
        PY=configured;
    elseif isfile(fallback)
        PY=fallback;
    else
        error(['No usable Python executable found. Pass it explicitly, e.g. ' ...
            'run_stageB_prime_all_seeds(''Python'',''/path/to/python3'').']);
    end
end
assert(isfile(PY),'Python executable does not exist: %s',PY);

outRoot=char(opt.OutputDir);
if ~isfolder(outRoot), mkdir(outRoot); end
logFile=fullfile(outRoot,'stageB_prime_run.log');
diary(logFile); cleanup=onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('Stage B'' multi-seed run started: %s\n',datetime('now'));
fprintf('Python: %s\nOutput: %s\nSeeds: %s\n',PY,outRoot,mat2str(opt.Seeds));

for seed=opt.Seeds(:)'
    dataDir=fullfile(outRoot,sprintf('data_seed%d',seed));
    outJson=fullfile(outRoot,sprintf('stageB_prime_seed%d.json',seed));
    if isfile(outJson)&&~opt.Force
        fprintf('\n[seed %d] Result exists; skipping (use Force=true to rerun).\n',seed);
        continue
    end
    fprintf('\n[seed %d] Generating data...\n',seed);
    run_stageB_prime_export(dataDir,seed);
    cmd=sprintf('"%s" "%s" --data "%s" --qsvm "%s" --out "%s" --seed %d --max-train %d', ...
        PY,pyDriver,dataDir,pyQsvm,outJson,seed,opt.MaxTrain);
    fprintf('[seed %d] Running Python experiment...\n',seed);
    [status,cmdout]=system(cmd);
    fprintf('%s\n',cmdout);
    if status~=0
        error('Stage B'' failed for seed %d (status %d). Command:\n%s\nOutput:\n%s',seed,status,cmd,cmdout);
    end
    assert(isfile(outJson),'Seed %d completed without producing %s',seed,outJson);
end

%% Aggregate every requested seed
rows=[];
for seed=opt.Seeds(:)'
    fn=fullfile(outRoot,sprintf('stageB_prime_seed%d.json',seed));
    assert(isfile(fn),'Missing result for seed %d: %s',seed,fn);
    J=jsondecode(fileread(fn));
    kinds={'summary','phase'}; conditions={'in_footprint','shifted_footprint','pmu_outage'};
    for ki=1:numel(kinds)
        kind=kinds{ki}; F=J.feature_sets.(kind);
        for ci=1:numel(conditions)
            cond=conditions{ci}; M=F.conditions.(cond).metrics;
            r=table(seed,string(kind),string(cond),F.n_train_used_all_models,F.n_qubits, ...
                M.LR.auc,M.RBF.auc,M.QSVM.auc,M.Hybrid.auc, ...
                M.QSVM.auc_difference_ci_vs_LR(1),M.QSVM.auc_difference_ci_vs_LR(2), ...
                M.QSVM.auc_difference_ci_vs_RBF(1),M.QSVM.auc_difference_ci_vs_RBF(2), ...
                F.hybrid_alpha, ...
                'VariableNames',{'Seed','FeatureSet','Condition','NTrain','NQubits', ...
                'AUC_LR','AUC_RBF','AUC_QSVM','AUC_Hybrid','QSVM_DiffLR_Lo','QSVM_DiffLR_Hi', ...
                'QSVM_DiffRBF_Lo','QSVM_DiffRBF_Hi','HybridAlpha'});
            rows=[rows;r]; %#ok<AGROW>
        end
    end
end
Results=rows;
Results.QSVMStrictWin=Results.AUC_QSVM>Results.AUC_LR & Results.AUC_QSVM>Results.AUC_RBF & ...
    Results.QSVM_DiffLR_Lo>0 & Results.QSVM_DiffRBF_Lo>0;
writetable(Results,fullfile(outRoot,'stageB_prime_all_seeds.csv'));
save(fullfile(outRoot,'stageB_prime_all_seeds.mat'),'Results','opt');

fprintf('\n================ MULTI-SEED SUMMARY ================\n');
target=Results.FeatureSet=="phase" & Results.Condition=="shifted_footprint";
T=Results(target,:); disp(T(:,{'Seed','AUC_LR','AUC_RBF','AUC_QSVM','QSVMStrictWin'}));
fprintf('Strict shifted-footprint QSVM wins: %d/%d seeds\n',sum(T.QSVMStrictWin),height(T));
fprintf('Mean AUC: LR=%.4f, RBF=%.4f, QSVM=%.4f\n',mean(T.AUC_LR),mean(T.AUC_RBF),mean(T.AUC_QSVM));
dLR=T.AUC_QSVM-T.AUC_LR; dRBF=T.AUC_QSVM-T.AUC_RBF;
[loLR,hiLR]=seed_boot_ci(dLR,10000,2026); [loRBF,hiRBF]=seed_boot_ci(dRBF,10000,2027);
fprintf('QSVM-LR:  mean=%+.4f, median=%+.4f, seed-bootstrap 95%% CI [%+.4f, %+.4f]\n',mean(dLR),median(dLR),loLR,hiLR);
fprintf('QSVM-RBF: mean=%+.4f, median=%+.4f, seed-bootstrap 95%% CI [%+.4f, %+.4f]\n',mean(dRBF),median(dRBF),loRBF,hiRBF);
ConfirmatorySummary=table(height(T),sum(T.QSVMStrictWin),mean(T.QSVMStrictWin), ...
    mean(T.AUC_LR),mean(T.AUC_RBF),mean(T.AUC_QSVM),mean(dLR),median(dLR),loLR,hiLR, ...
    mean(dRBF),median(dRBF),loRBF,hiRBF, ...
    'VariableNames',{'NSeeds','NStrictWins','StrictWinRate','MeanAUC_LR','MeanAUC_RBF','MeanAUC_QSVM', ...
    'MeanDiff_LR','MedianDiff_LR','BootLo_LR','BootHi_LR','MeanDiff_RBF','MedianDiff_RBF','BootLo_RBF','BootHi_RBF'});
writetable(ConfirmatorySummary,fullfile(outRoot,'stageB_prime_confirmatory_summary.csv'));
save(fullfile(outRoot,'stageB_prime_confirmatory_summary.mat'),'ConfirmatorySummary','T','dLR','dRBF');
fprintf('Saved aggregate CSV and MAT files in %s\n',outRoot);
end

function [lo,hi]=seed_boot_ci(d,B,seed)
% Percentile bootstrap of the mean paired AUC difference across seeds.
rng(seed); d=d(:); n=numel(d); boot=zeros(B,1);
for b=1:B, boot(b)=mean(d(randi(n,n,1))); end
q=quantile(boot,[.025 .975]); lo=q(1); hi=q(2);
end
