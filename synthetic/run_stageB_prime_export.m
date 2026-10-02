function run_stageB_prime_export(outdir, seed)
% Export leakage-safe raw feature tables for the complete Stage B' test.
% Label: 1=true FO, 0=GPS-spoofed FO-like event.
if nargin<1 || isempty(outdir), outdir='stageB_prime_data'; end
if nargin<2 || isempty(seed), seed=42; end
close all; clc; rng(seed);
if ~isfolder(outdir), mkdir(outdir); end

N=12; fs=60; T=20; band=[0.1 2.5]; nTrain=2000; nTest=600;
regTrain=1:6; regShift=7:12;
fprintf('Generating Stage B'' seed %d data...\n',seed);
[trX,trY]=gen_data(nTrain,N,fs,T,band,regTrain);
[inX,inY]=gen_data(nTest,N,fs,T,band,regTrain);
[shX,shY]=gen_data(nTest,N,fs,T,band,regShift);
[ouX,ouY]=gen_data(nTest,N,fs,T,band,regTrain);
mask=true(1,N); mask(randperm(N,3))=false;

write_set(outdir,'train', summary_feat(trX,fs,band), phase_feat(trX,fs,band,N),trY);
write_set(outdir,'in_footprint',summary_feat(inX,fs,band),phase_feat(inX,fs,band,N),inY);
write_set(outdir,'shifted_footprint',summary_feat(shX,fs,band),phase_feat(shX,fs,band,N),shY);
write_set(outdir,'pmu_outage',summary_mask(ouX,fs,band,mask),phase_mask(ouX,fs,band,mask,N),ouY);
meta=struct('seed',seed,'n_pmus',N,'fs',fs,'window_seconds',T,'fo_band_hz',band, ...
 'n_train',nTrain,'n_test',nTest,'outage_mask',mask,'spoof_region_train',regTrain,'spoof_region_shift',regShift);
fid=fopen(fullfile(outdir,'metadata.json'),'w'); fprintf(fid,'%s',jsonencode(meta,'PrettyPrint',true)); fclose(fid);
fprintf('Exported raw features to %s\n',outdir);
end

function write_set(d,n,S,P,y)
writematrix([S y],fullfile(d,[n '_summary.csv']));
writematrix([P y],fullfile(d,[n '_phase.csv']));
end

function [X,y]=gen_data(nw,N,fs,T,band,region)
L=round(T*fs); t=(0:L-1)'/fs; X=cell(nw,1); y=zeros(nw,1);
for k=1:nw
 isTrue=rand<.5; y(k)=isTrue; df=.02*randn(L,N); v=1+.005*randn(L,N);
 f0=band(1)+diff(band)*rand; A=.008+.012*rand; base=2*pi*rand;
 for n=1:N
  if isTrue, ph=base+deg2rad(8)*randn;
  elseif ismember(n,region)
   pos=(find(region==n,1)-1)/max(1,numel(region)-1);
   ph=base+deg2rad(60)*(pos-.5)+deg2rad(35)*randn;
  else, ph=base+deg2rad(8)*randn; end
  df(:,n)=df(:,n)+A*sin(2*pi*f0*t+ph); v(:,n)=v(:,n)+.05*A*sin(2*pi*f0*t+ph);
 end
 X{k}=struct('df',df,'v',v);
end
end

function F=summary_feat(X,fs,band)
n=numel(X); F=zeros(n,8); bp=designfilt('bandpassiir','FilterOrder',4,'HalfPowerFrequency1',band(1),'HalfPowerFrequency2',band(2),'SampleRate',fs);
for i=1:n
 df=X{i}.df; v=X{i}.v; [L,N]=size(df); vb=filtfilt(bp,v); db=filtfilt(bp,df);
 vr=sqrt(mean(vb.^2)); dr=sqrt(mean(db.^2)); Vs=mean(v,2); Y=abs(fft(Vs)); f=(0:L-1)*(fs/L);
 [~,km]=max(Y(2:floor(L/2))); km=km+1; ph=angle(fft(v)); ph=ph(km,:);
 wl=min(128,floor(L/2)); no=floor(.5*wl); nf=max(256,2^nextpow2(wl)); pr=nchoosek(1:N,2); cf=zeros(size(pr,1),1);
 for p=1:size(pr,1), [C,ff]=mscohere(vb(:,pr(p,1)),vb(:,pr(p,2)),hamming(wl),no,nf,fs); cf(p)=mean(C(ff>=band(1)&ff<=band(2))); end
 P=abs(fft(Vs(1:floor(L/2)))).^2+eps;
 F(i,:)=[max(abs(df),[],'all'),max(vr),mean(vr),max(dr),f(km),1-abs(mean(exp(1j*ph))),mean(cf),exp(mean(log(P)))/mean(P)];
end
end

function F=summary_mask(X,fs,band,m)
for i=1:numel(X), X{i}.v=X{i}.v(:,m); X{i}.df=X{i}.df(:,m); end; F=summary_feat(X,fs,band);
end

function F=phase_feat(X,fs,band,Nfix)
n=numel(X); F=zeros(n,2*Nfix+4); bp=designfilt('bandpassiir','FilterOrder',4,'HalfPowerFrequency1',band(1),'HalfPowerFrequency2',band(2),'SampleRate',fs);
for i=1:n
 v=X{i}.v; [L,N]=size(v); vb=filtfilt(bp,v); Y=abs(fft(mean(vb,2))); [~,km]=max(Y(2:floor(L/2))); km=km+1;
 V=fft(vb); ph=angle(V(km,:)); mp=angle(mean(exp(1j*ph))); rel=angle(exp(1j*(ph-mp)));
 c=zeros(1,Nfix); s=c; c(1:N)=cos(rel); s(1:N)=sin(rel);
 wl=min(128,floor(L/2)); no=floor(.5*wl); nf=max(256,2^nextpow2(wl)); pr=nchoosek(1:N,2); cf=zeros(size(pr,1),1);
 for p=1:size(pr,1), [C,ff]=mscohere(vb(:,pr(p,1)),vb(:,pr(p,2)),hamming(wl),no,nf,fs); cf(p)=mean(C(ff>=band(1)&ff<=band(2))); end
 F(i,:)=[c s 1-abs(mean(exp(1j*ph))) std(sort(rel)) max(abs(rel)) mean(cf)];
end
end

function F=phase_mask(X,fs,band,m,Nfix)
for i=1:numel(X), X{i}.v=X{i}.v(:,m); X{i}.df=X{i}.df(:,m); end; F=phase_feat(X,fs,band,Nfix);
end
