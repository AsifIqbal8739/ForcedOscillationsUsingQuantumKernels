function Summary = make_revision_figures_consistent(stageBMat, finalJson, featureJson, outDir)
% Publication figures for the revised QSVM/FO manuscript.
% Example:
% make_revision_figures('stageB_prime_confirmatory/stageB_prime_all_seeds.mat', ...
%   'isone_final_matched.json','isone_nonstationary.json','revision_figures');
if nargin<4||isempty(outDir),outDir='revision_figures';end
if ~isfolder(outDir),mkdir(outDir);end
S=load(stageBMat); assert(isfield(S,'Results'),'MAT file must contain Results table'); T=S.Results;
J=jsondecode(fileread(finalJson)); F=jsondecode(fileread(featureJson));
cols=[0.20 0.45 0.75;0.85 0.33 0.10;0.25 0.65 0.35]; names={'LR','RBF-SVM','QSVM'};
set(groot,'defaultAxesFontName','Times New Roman','defaultAxesFontSize',9,'defaultTextFontName','Times New Roman');

%% Fig. 1 - Confirmatory synthetic spatial-shift result
Q=T(T.FeatureSet=="phase" & T.Condition=="shifted_footprint",:);
f=figure('Color','w','Position',[80 80 1000 390]);tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; hold on
for i=1:height(Q),plot(1:3,[Q.AUC_LR(i),Q.AUC_RBF(i),Q.AUC_QSVM(i)],'-','Color',[.78 .78 .78],'LineWidth',.6);end
for m=1:3
 v={Q.AUC_LR,Q.AUC_RBF,Q.AUC_QSVM}; scatter(m+zeros(height(Q),1),v{m},22,cols(m,:),'filled','MarkerFaceAlpha',.72);
 [mu,lo,hi]=bootmean(v{m},10000,500+m);errorbar(m,mu,mu-lo,hi-mu,'k','LineWidth',1.4,'CapSize',8);
end
xlim([.55 3.45]);ylim([0.3 1]);xticks(1:3);xticklabels(names);ylabel('AUC');title('(a) Frozen 20-seed confirmation');grid on;box on
nexttile;hold on;d1=Q.AUC_QSVM-Q.AUC_LR;d2=Q.AUC_QSVM-Q.AUC_RBF;
boxchart([ones(size(d1));2*ones(size(d2))],[d1;d2],'BoxFaceColor',[.45 .45 .75]);yline(0,'k--');
% Use the same bootstrap seeds as run_stageB_prime_all_seeds.m so the
% plotted confidence intervals exactly match the abstract and summary CSV.
ciSeeds=[2026 2027];
for k=1:2
 d={d1,d2};[mu,lo,hi]=bootmean(d{k},10000,ciSeeds(k));errorbar(k,mu,mu-lo,hi-mu,'r','LineWidth',1.4,'CapSize',8);
end
xticks(1:2);xticklabels({'QSVM - LR','QSVM - RBF'});ylabel('Delta AUC');title('(b) Paired seed-level advantage');grid on;box on
export_all(f,outDir,'Fig_synthetic_confirmation');

%% Convert final JSON into one row per seed/case/model
[A,seed,cas]=matched_arrays(J); % nRun x 5 x 3

%% Fig. 2 - ISO-NE external validation by event
f=figure('Color','w','Position',[80 80 1050 430]);hold on
mu=squeeze(mean(A,1));lo=zeros(5,3);hi=lo;
for c=1:5,for m=1:3,[~,lo(c,m),hi(c,m)]=bootmean(A(:,c,m),10000,700+10*c+m);end,end
b=bar(1:5,mu,'grouped');for m=1:3,b(m).FaceColor=cols(m,:);end
for m=1:3,x=b(m).XEndPoints;errorbar(x,mu(:,m),mu(:,m)-lo(:,m),hi(:,m)-mu(:,m),'k.','LineWidth',1,'CapSize',5);end
ylim([0.35 1.03]);xticks(1:5);xticklabels(compose('Case %d',1:5));ylabel('Event-held-out AUC');
legend(names,'Location','southoutside','Orientation','horizontal');title('Real ISO-NE records with semi-synthetic timing attacks');grid on;box on
export_all(f,outDir,'Fig_ISONE_matched_AUC');

%% Fig. 3 - Case-wise QSVM differences and confidence intervals
f=figure('Color','w','Position',[80 80 930 430]);hold on;y=1:5;
for c=1:5
 dLR=A(:,c,3)-A(:,c,1);dR=A(:,c,3)-A(:,c,2);
 [m1,l1,h1]=bootmean(dLR,10000,800+c);[m2,l2,h2]=bootmean(dR,10000,900+c);
 errorbar(m1,c-.11,m1-l1,h1-m1,'horizontal','LineStyle','none','Marker','o', ...
     'Color',cols(1,:),'MarkerFaceColor',cols(1,:),'CapSize',6);
 errorbar(m2,c+.11,m2-l2,h2-m2,'horizontal','LineStyle','none','Marker','s', ...
     'Color',cols(2,:),'MarkerFaceColor',cols(2,:),'CapSize',6);
end
xline(0,'k--');yticks(y);yticklabels(compose('Case %d',1:5));set(gca,'YDir','reverse');xlabel('Delta AUC relative to classical baseline');
legend({'QSVM - LR','QSVM - RBF'},'Location','southoutside','Orientation','horizontal');title('External-validation effect sizes (seed-bootstrap 95% CI)');grid on;box on
export_all(f,outDir,'Fig_ISONE_effect_sizes');

%% Fig. 4 - Representation ablation
[B,repNames]=feature_arrays(F); % seed x case x representation x model
overall=squeeze(mean(B,[1 2]));
f=figure('Color','w','Position',[80 80 1050 430]);tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;b=bar(overall,'grouped');for m=1:3,b(m).FaceColor=cols(m,:);end
xticklabels(repNames);ylabel('Mean event-held-out AUC');ylim([.45 .9]);title('(a) Overall representation ablation');grid on;box on
nexttile;qs=squeeze(mean(B(:,:,:,3),1));bar(qs,'grouped');xticks(1:5);xticklabels(compose('Case %d',1:5));
ylabel('QSVM AUC');ylim([.35 1]);legend(repNames,'Location','southoutside','Orientation','horizontal');title('(b) QSVM sensitivity by event');grid on;box on
export_all(f,outDir,'Fig_feature_ablation');

%% Fig. 5 - Hyperparameter-selection transparency
f=figure('Color','w','Position',[80 80 1000 390]);tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
Cs=zeros(numel(seed)*5,3);beta=zeros(numel(seed)*5,1);bp=beta;k=0;
for r=1:numel(J.runs),for c=1:numel(J.runs(r).folds),k=k+1;M=J.runs(r).folds(c).models;
 Cs(k,:)=[M.LR.selected.C,M.RBF.selected.C,M.QSVM.selected.C];beta(k)=M.QSVM.selected.beta;bp(k)=M.QSVM.selected.bp;end,end
nexttile;bar(categorical(string([.1 1 10])),[sum(Cs==.1);sum(Cs==1);sum(Cs==10)]);title('(a) Selected C');ylabel('Outer folds');legend(names,'Location','best');grid on
nexttile;histogram(categorical(string(beta)));title('(b) QSVM rotation scale \beta');ylabel('Outer folds');grid on
nexttile;histogram(categorical(string(bp)));title('(c) QSVM interaction scale \beta''');ylabel('Outer folds');grid on
export_all(f,outDir,'Fig_hyperparameter_selection');

Summary=table(mean(Q.AUC_LR),mean(Q.AUC_RBF),mean(Q.AUC_QSVM),mean(d1),mean(d2), ...
 mean(A(:,:,1),'all'),mean(A(:,:,2),'all'),mean(A(:,:,3),'all'), ...
 'VariableNames',{'Synthetic_LR','Synthetic_RBF','Synthetic_QSVM','Synthetic_DiffLR','Synthetic_DiffRBF','ISONE_LR','ISONE_RBF','ISONE_QSVM'});
writetable(Summary,fullfile(outDir,'figure_summary_values.csv'));
save(fullfile(outDir,'figure_source_data.mat'),'Q','A','B','Summary','seed','cas');
end

function [A,seeds,cases]=matched_arrays(J)
nr=numel(J.runs);A=nan(nr,5,3);seeds=zeros(nr,1);cases=1:5;
for r=1:nr,seeds(r)=J.runs(r).seed;for c=1:numel(J.runs(r).folds),z=J.runs(r).folds(c);i=z.heldout_case;A(r,i,:)=[z.models.LR.auc,z.models.RBF.auc,z.models.QSVM.auc];end,end
end
function [B,names]=feature_arrays(F)
names={'Static','Multiscale','Structured'};nr=numel(F.runs);B=nan(nr,5,3,3);
for r=1:nr,for c=1:numel(F.runs(r).folds),z=F.runs(r).folds(c);i=z.heldout_case;
 for q=1:3,x=z.representations.(lower(names{q}));B(r,i,q,:)=[x.auc.LR,x.auc.RBF,x.auc.QSVM];end,end,end
end
function [mu,lo,hi]=bootmean(x,B,seed)
rng(seed);x=x(:);n=numel(x);z=zeros(B,1);for b=1:B,z(b)=mean(x(randi(n,n,1)));end;mu=mean(x);q=quantile(z,[.025 .975]);lo=q(1);hi=q(2);
end
function export_all(f,d,n)
exportgraphics(f,fullfile(d,[n '.pdf']),'ContentType','vector');exportgraphics(f,fullfile(d,[n '.png']),'Resolution',600);savefig(f,fullfile(d,[n '.fig']));
end