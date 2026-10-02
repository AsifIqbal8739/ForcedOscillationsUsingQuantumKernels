function run_stageB_ablation_spoof(varargin)
%% ============ Stage B' MECHANISM ABLATION: is the win from physics-wired entanglement? ============
% Stage B' established a replicated, significant QSVM > RBF > LR advantage
% on the shifted-footprint true-vs-spoofed FO attribution task, using
% permutation-invariant phase-geometry features (INV). This ablation asks
% the load-bearing question for the paper's central claim:
%
%   Is that advantage attributable to the PHYSICS-INFORMED entangling
%   structure, or merely to using a quantum kernel at all?
%
% It compares, on the SAME shifted-spoof task, across multiple seeds:
%   physics : ZZ entanglers wired to physically-related INV feature pairs
%   none    : no entangling gates (pure product-state kernel)
%   ring    : generic hardware-efficient ring entanglement (physics-agnostic)
%
% Decision:
%   If physics significantly beats BOTH none and ring (seed-bootstrap CI on
%   the paired AUC gap excludes 0), the "physics-informed" claim is earned.
%   If physics ties none/ring, the honest claim shrinks to "a quantum kernel
%   helps here" -- the entangling structure is not the mechanism.
%
% Requires qsvm_kernel_v2.py (with --edges support) next to this script.
%
% Usage: run_stageB_ablation_spoof              (default seeds 200:219)
%        run_stageB_ablation_spoof(200:204)     (custom seed vector)

if nargin>=1 && ~isempty(varargin{1}), seeds=varargin{1}; else, seeds=200:219; end

N_PMUs=12; fs=60; T_win_sec=20; foFreqHz=[0.1 2.5];
nTrain=800; nTest=600;               % nTrain>400 for tighter per-seed AUCs
spoof_region_train=1:6; spoof_region_shift=7:12;

PY=string(pyenv().Executable); PY_QSVM="qsvm_kernel_v2.py";
assert(isfile(PY_QSVM),'Put qsvm_kernel_v2.py next to this script.');
tmpdir=tempname; mkdir(tmpdir);

% ---- INV feature layout (must match feat_inv below) ----
% [disp_circ, grad, maxlag, q1..q5 (sorted |rel-phase| quantiles), mean_coh, max_coh]
% dim = 3 + 5 + 2 = 10 features -> 10 qubits.
% PHYSICS-WIRED edges: connect features that are physically related --
%   disp_circ(0)<->grad(1)      : two global dispersion descriptors
%   maxlag(2)<->q5(7)           : extreme-phase descriptors (max lag ~ top quantile)
%   q1(3)<->q2(4), q3(5)<->q4(6): adjacent quantiles of the phase-magnitude profile
%   mean_coh(8)<->max_coh(9)    : the two coherence descriptors
PHYS_EDGES = '0-1,2-7,3-4,5-6,8-9';
% generic ring over 10 qubits
RING_EDGES = '0-1,1-2,2-3,3-4,4-5,5-6,6-7,7-8,8-9,9-0';

topos = {'physics', PHYS_EDGES; 'none', 'none'; 'ring', RING_EDGES};

fprintf('Stage B'' ablation: physics vs none vs ring, on shifted-spoof, seeds %d..%d\n', seeds(1), seeds(end));

AUC = struct('physics',zeros(numel(seeds),1),'none',zeros(numel(seeds),1),'ring',zeros(numel(seeds),1));

for si=1:numel(seeds)
    sd=seeds(si); rng(sd);
    % data
    [trX,trY]=gen_true_vs_spoof(nTrain,N_PMUs,fs,T_win_sec,foFreqHz,spoof_region_train);
    [teX,teY]=gen_true_vs_spoof(nTest,N_PMUs,fs,T_win_sec,foFreqHz,spoof_region_shift); % SHIFTED
    % INV features
    trF=feat_inv(trX,fs,foFreqHz); teF=feat_inv(teX,fs,foFreqHz);
    [trFz,mu,sg]=zscore(trF); sg(~isfinite(sg)|sg==0)=1e-9; teFz=(teF-mu)./sg;
    trFz(~isfinite(trFz))=0; teFz(~isfinite(teFz))=0;
    scale=1000; gain=1000; ang=@(F)scale*tanh(F/gain);
    trc=fullfile(tmpdir,sprintf('tr_%d.csv',sd)); writematrix([ang(trFz) trY],trc);
    tec=fullfile(tmpdir,sprintf('te_%d.csv',sd)); writematrix([ang(teFz) teY],tec);

    % CV hyperparameters ONCE per seed (shared across topologies for fairness:
    % we vary ONLY the edges, holding beta/beta'/C at the physics-CV choice, so
    % any difference is attributable to topology, not to re-tuned scales).
    ojcv=fullfile(tmpdir,sprintf('cv_%d.json',sd));
    cmd=sprintf('"%s" "%s" --train "%s" --test "%s" --out "%s" --edges "%s"', ...
        PY,PY_QSVM,trc,tec,ojcv,PHYS_EDGES);
    assert(system(cmd)==0,'CV failed'); Scv=jsondecode(fileread(ojcv));
    HP=sprintf('--no_cv --beta %.6g --beta_prime %.6g --C %.6g', ...
        Scv.hyperparams.beta,Scv.hyperparams.beta_prime,Scv.hyperparams.C);

    for ti=1:size(topos,1)
        tname=topos{ti,1}; edges=topos{ti,2};
        oj=fullfile(tmpdir,sprintf('%s_%d.json',tname,sd));
        cmd=sprintf('"%s" "%s" --train "%s" --test "%s" --out "%s" --edges "%s" %s', ...
            PY,PY_QSVM,trc,tec,oj,edges,HP);
        assert(system(cmd)==0,sprintf('QSVM %s failed seed %d',tname,sd));
        S=jsondecode(fileread(oj)); prob=double(S.pred_probs(:));
        [~,~,~,a]=perfcurve(teY,prob,1);
        AUC.(tname)(si)=a;
    end
    fprintf('  seed %d: physics=%.4f none=%.4f ring=%.4f\n', sd, ...
        AUC.physics(si), AUC.none(si), AUC.ring(si));
end

%% ---- paired seed-bootstrap analysis ----
d_phys_none = AUC.physics - AUC.none;
d_phys_ring = AUC.physics - AUC.ring;
[m1,lo1,hi1] = seed_bootstrap(d_phys_none, 5000, 7);
[m2,lo2,hi2] = seed_bootstrap(d_phys_ring, 5000, 7);

fprintf('\n================================ ABLATION SUMMARY ================================\n');
fprintf('Mean AUC:  physics=%.4f   none=%.4f   ring=%.4f   (over %d seeds)\n', ...
    mean(AUC.physics), mean(AUC.none), mean(AUC.ring), numel(seeds));
fprintf('physics - none:  mean=%+.4f  seed-bootstrap 95%% CI [%+.4f, %+.4f]\n', m1, lo1, hi1);
fprintf('physics - ring:  mean=%+.4f  seed-bootstrap 95%% CI [%+.4f, %+.4f]\n', m2, lo2, hi2);
n_phys_best = sum(AUC.physics>=AUC.none & AUC.physics>=AUC.ring);
fprintf('physics is best (>= both) on %d/%d seeds\n', n_phys_best, numel(seeds));

fprintf('\n================================ VERDICT ================================\n');
phys_beats_none = lo1 > 0;
phys_beats_ring = lo2 > 0;
if phys_beats_none && phys_beats_ring
    fprintf(['>>> PHYSICS-WIRED ENTANGLEMENT IS THE MECHANISM: physics significantly beats\n' ...
             '    BOTH none and ring (both seed-bootstrap CIs exclude 0). The paper CAN claim\n' ...
             '    a physics-informed quantum kernel advantage -- the entangling structure,\n' ...
             '    not merely "a quantum kernel", drives the effect. (Recommend one fresh-seed\n' ...
             '    confirmation block before finalizing.)\n']);
elseif ~phys_beats_none && ~phys_beats_ring
    fprintf(['>>> ENTANGLING STRUCTURE IS NOT THE MECHANISM: physics ties none and ring.\n' ...
             '    The quantum-kernel advantage on this task is real but is NOT attributable to\n' ...
             '    the physics-informed wiring. Honest claim shrinks to "a quantum kernel helps\n' ...
             '    on this shifted-spoof task"; the paper must NOT claim the physics wiring is\n' ...
             '    the cause. (This matches every prior ablation in the investigation.)\n']);
else
    fprintf(['>>> PARTIAL: physics beats one of {none,ring} but not both. Ambiguous --\n' ...
             '    physics-beats-none=%d, physics-beats-ring=%d. Report exactly which, and treat\n' ...
             '    the physics-informed claim as suggestive-not-established pending more seeds.\n'], ...
             phys_beats_none, phys_beats_ring);
end
fprintf('=========================================================================\n');

save('stageB_ablation_results.mat','AUC','seeds','m1','lo1','hi1','m2','lo2','hi2');
fprintf('\nSaved to stageB_ablation_results.mat\n');

end  % ===== end main =====


%% ===================== helpers =====================

function [m,lo,hi]=seed_bootstrap(d, B, seed)
    rng(seed); d=d(:); n=numel(d); bs=zeros(B,1);
    for b=1:B, bs(b)=mean(d(randi(n,n,1))); end
    m=mean(d); lo=quantile(bs,0.025); hi=quantile(bs,0.975);
end

function [X,y]=gen_true_vs_spoof(nWin,N_PMUs,fs,T_sec,foBandHz,spoof_region)
    FOAmpRange=[0.008 0.02]; TruePhaseJitter=deg2rad(8);
    SpoofExtraDisp=deg2rad(35); SpoofGradMax=deg2rad(60);
    AmbientDfStd=0.02; AmbientVStd=0.005;
    L=round(T_sec*fs); t=(0:L-1)'/fs; X=cell(nWin,1); y=zeros(nWin,1);
    for k=1:nWin
        isTrue=rand<0.5; y(k)=double(isTrue);
        df=AmbientDfStd*randn(L,N_PMUs); v=1+AmbientVStd*randn(L,N_PMUs);
        f0=foBandHz(1)+(foBandHz(2)-foBandHz(1))*rand; A=FOAmpRange(1)+(FOAmpRange(2)-FOAmpRange(1))*rand;
        base=2*pi*rand();
        for n=1:N_PMUs
            if isTrue, ph=base+TruePhaseJitter*randn;
            else
                if ismember(n,spoof_region)
                    pos=(find(spoof_region==n)-1)/max(1,(numel(spoof_region)-1));
                    ph=base+SpoofGradMax*(pos-0.5)+SpoofExtraDisp*randn;
                else, ph=base+TruePhaseJitter*randn; end
            end
            df(:,n)=df(:,n)+A*sin(2*pi*f0*t+ph); v(:,n)=v(:,n)+0.05*A*sin(2*pi*f0*t+ph);
        end
        Xi.df=df; Xi.v=v; X{k}=Xi;
    end
end

function F=feat_inv(Xcell,fs,foBand)
% Permutation-invariant phase-geometry features. MUST match the qubit/edge
% layout documented at the top (10 features): [disp_circ, grad, maxlag,
% q1..q5, mean_coh, max_coh].
    n=numel(Xcell); nq=5; F=zeros(n, 3+nq+2);
    bp=designfilt('bandpassiir','FilterOrder',4,'HalfPowerFrequency1',foBand(1),'HalfPowerFrequency2',foBand(2),'SampleRate',fs);
    for i=1:n
        v=Xcell{i}.v; [L,Nn]=size(v);
        vb=filtfilt(bp,v);
        Vs=mean(vb,2); Y=abs(fft(Vs)); [~,km]=max(Y(2:floor(L/2))); km=km+1;
        Vk=fft(vb); Vk=Vk(km,:); ph=angle(Vk);
        mean_ph=angle(mean(exp(1j*ph))); rel=angle(exp(1j*(ph-mean_ph)));
        disp_circ=1-abs(mean(exp(1j*ph))); grad=std(sort(rel)); maxlag=max(abs(rel));
        q=quantile(abs(rel),linspace(0,1,nq));
        wl=min(128,floor(L/2)); no=floor(0.5*wl); nf=max(256,2^nextpow2(wl));
        if Nn>1 && L>wl+2
            pr=nchoosek(1:Nn,2); cf=zeros(size(pr,1),1);
            for p=1:size(pr,1)
                [Cxy,ff]=mscohere(vb(:,pr(p,1)),vb(:,pr(p,2)),hamming(wl),no,nf,fs);
                m=ff>=foBand(1)&ff<=foBand(2); cf(p)=mean(Cxy(m));
            end
            mc=[mean(cf) max(cf)];
        else, mc=[0 0]; end
        F(i,:)=[disp_circ,grad,maxlag,q,mc];
    end
end
