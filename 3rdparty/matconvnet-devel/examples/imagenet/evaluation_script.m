opts.dataDir = '/home/amirro/storage/data/ILSVRC2012_pre/';
% opts.modelType = 'alexnet' ;
opts.networkType = 'simplenn' ;
opts.expDir = '/home/amirro/storage/data/imagenet12-alexnet-bnorm-simplenn';
opts.imdbPath = fullfile(opts.expDir,'imdb.mat');
opts.modelPath = fullfile(opts.expDir,'net-epoch-1.mat');

imdb = load(opts.imdbPath);
% net = load(opts.modelPath);
% net = net.net;
% net = cnn_imagenet_deploy(net);

run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath(genpath('~/code/utils'));
rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');

%%



% [recalls,precisions,F_scores,per_class_losses]  = doEvaluation( opts,imdb,1,40,'');
nEpochs=20;
[all_results,all_labels]  = getAllResults( opts,imdb,1,nEpochs,'',2 );
[all_results_1,all_results_2] = splitSets(all_results,2);
[all_labels_1,all_labels_2] = splitSets(all_labels,1);


[F_scores_1,cms_1,perfs_1,precisions_1,recalls_1,class_losses_1] = aggregateMetrics(all_results_1,all_labels_1,1000);
[F_scores_2,cms_2,perfs_2,precisions_2,recalls_2,class_losses_2] = aggregateMetrics(all_results_2,all_labels_2,1000);
figure(1);
showPerformanceAnalysis(perfs_1,nEpochs,F_scores_1);
figure(2);
showPerformanceAnalysis(perfs_2,nEpochs,F_scores_2);

% make a score ensemble
% 1. without normalization
[m1,im1] = max(F_scores_1,[],2);
[m2,im2] = max(F_scores_2,[],2);

final_f_scores = F_scores_1(:,end);
[r,ir] = sort(final_f_scores,'ascend');
%%
for it = 1:length(ir)
    t = ir(it);
    f1 = F_scores_1(t,:);
%     f = f-min(f);
%     f = f/max(f);
    figure(1),clf;plot(f);
    f2 = class_losses_1(t,:);
%     f = f-min(f);
%     f = f/max(f);hold on; plot(f); 
    x = 1:length(f1);
    plotyy(x,f1,x,f2);
    legend('f-scores','losses');
    title({imdb.classes.description{t},num2str(F_scores_1(t,end))});
    dpc;
end
%%
% plot the recall of this class.
plot(precisions_1(sel_,:),recalls_1(sel_,:),'-+')
F_scores_1(sel_,:)

imdb.classes.description(ir(end:-1:end-9))'


recalls(sel_,:)
precisions(se



[v,iv] = sort(im1);

for it = 1:length(iv)
    it
    t = iv(it);
    clf; plot(F_scores_1(t,:));
    dpc
end

means = zeros(nEpochs,1);
counts = zeros(nEpochs,1);
for t = 1:nEpochs   
    
    means(t) = mean(m1(im1==t));
    counts(t) = length(m1(im1==t));
end

boxplot(F_scores_1(:,end),im1)

figure,bar(counts);
hold on;plot(1:40,100*means,'r-d');
plot(sort(F_scores_1(:,end)))
figure,plot(m1,m2,'r+');
figure,hist(double(im1-im2));

% corr(m1,m2)
% figure,plot(m1); hold on; plot(m2)

% std(im1-im2)
nClasses = 1000;
all_results_softmaxed2 = {};
for t = 1:length(all_results)
    t
    curRes = reshape(all_results_2{t}, 1,1,nClasses,[]);
    all_results_softmaxed2{t} = squeeze(vl_nnsoftmax(curRes));
end

%%

fprintf('trying to combine softmaxed results...\n');
newScores = zeros(size(all_results_2{1}),'single');
for t = 1:nClasses
    t
    curBest = im1(t);
    newScores(t,:) = all_results_softmaxed2{curBest}(t,:);
end
% all(col(all_labels{t}==all_labels{t-1}))
[~,cm_new,precision_new,recall_new,F_score_new] = evaluationMetrics(newScores,all_labels_2{1},1000);
perf_new = sum(diag(cm_new))/sum(cm_new(:));
fprintf('improvement (validation): %%%2.2f\n',100*(perf_new-perfs_2(end)));
%%
%
N = 1000;

plot(F_sc

all_new_F_scores = zeros(size(F_scores_2));
newScores_test = zeros(size(all_results_2{1}));
for k = 1:40
    %     k
    
    %     for m = nEpochs-k+1:nEpochs
    %newScores_test = newScores_test + all_results_softmaxed2{end-k+1};
    newScores_test = newScores_test + all_results_softmaxed2{end-k+1};
    %     end    
    % all(col(all_labels{t}==all_labels{t-1}))
    [~,cm_new,precision_new_test,recall_newtest,F_score_new_test] = evaluationMetrics(newScores_test,all_labels_2{1},N);         
    all_new_F_scores(:,k) = F_score_new_test;
    perf_new_test = sum(diag(cm_new))/sum(cm_new(:));
    fprintf('improvement(test, avg of last %d): %%%2.2f\n',k,100*(perf_new_test-perfs_2(end)));
end





% % imdb.images.name(imdb.images.set==3)
% for eee = 1:40
%     eee
%     [all_results,all_labels]  = getAllResults_batches( opts,imdb,eee,eee,'_test_frac10',3,10 );
% end


%%
[per_class_losses,cm,precision,recall,F_score] = evaluationMetrics(results,labels,N)
%
save(fullfile(perfDir,'perf_summary_normal_40.mat'),'recalls','precisions','F_scores','per_class_losses');
F_scores2 = F_scores;
F_scores2(isnan(F_scores2)) = 0;

%% instead of F_scores, calculate accuracies.
% F_scores2 = class_accuracies;

figure,plot(mean(F_scores2));

boxplot(F_scores2);

% aucs_bkp = aucs;
aucs = F_scores2;
aucs_n = bsxfun(@rdivide,aucs,max(aucs,[],2));
aucs_n(isnan(aucs_n)) = 0;
%figure,plot(mean(aucs_n))
% find out the early and late bloomers
nEpochs=40;
areas = zeros(1000,1);
for t = 1:1000
    p = 1:nEpochs;
    r = aucs_n(t,:);
    areas(t)=trapz(p,r);
end


%%
[v,iv] = max(F_scores2,[],2);
figure,hist(iv);


%%
[v,iv] = sort(areas,'ascend');

for it = 1:10:length(iv)
    if (v(it)==0)
        continue
    end
    t = iv(it)
    y = aucs_n(t,:);
    clf;
    plot(y,'-o'); title(sprintf('class:%s , max score: %f ', imdb.classes.description{t}, max(aucs(t,:))));
    hold on; plot(aucs(t,:),'-o');
    h = legend('normalized','original');
    rect = [.6 .2 .2 .2];
    set(h,'position',rect);
    dpc
end
%% plot the f measures sorted according to largest gap between best measure
% and final measure:
% [v,iv] = sort(F_scores(:,end),'ascend');
F_scores2 = F_scores;
% F_scores2 = -per_class_losses;
% F_scores2 = precisions;
final_f = F_scores2(:,end);
[best_f,ibest] = max(F_scores2,[],2);
ff = final_f-best_f;
[v,iv] = sort(ff,'ascend');

% hist(ibest,1:20)
% [v,iv] = sort(ibest,'ascend');
% hist(ibest,1:20)

% figure,hist(best_f./final_f)
for it = 1:1:1000
    t = iv(it)
    y = F_scores2(t,:);
    clf;
    plot(y,'-o'); title(sprintf('class:%s , max score: %f , final score: %f ', imdb.classes.description{t}, max(y),y(end)));
    zz = max(y);
    hold on;plot(1:length(y),ones(size(y))*zz);
    %     y2 = 100*F_scores(t,:);
    %     y2 = y2-mean(y2)+mean(y);
    %     plot(y2,'r-o');
    %     ylim([0 1]);
    dpc;
    continue
    hold on; plot(aucs(t,:),'-o');
    h = legend('normalized','original');
    rect = [.6 .2 .2 .2];
    set(h,'position',rect);
    dpc
end
%plot(ibest,best_f,'r.');
%boxplot(best_f,ibest)



%% postulate a correlation between weak classes and those where there was a large gap
figure(5),clf;plot(ff,final_f,'r+');
xlabel('final perf - best perf');
ylabel('best perf');
title('Final/Best perfomance gap vs. overall performance')
gcf
saveas(gcf,'/home/amirro/notes/images/helping_the_weak/gap_vs_best.fig');
%% % show some of the images for the weak class, 999
batch = find(imdb.images.label == 999);
[im, labels] = getBatch(imdb, batch) ;

for t = 1:10:1000
    clf; imagesc2(im(:,:,:,t)/255+.5); dpc
end


%% now, having re-trained using only the weakest class, let's check what happened

[recalls_orig,precisions_orig,F_scores_orig,per_class_losses_orig] = deal(recalls,precisions,F_scores,per_class_losses);

[recalls_weak,precisions_weak,F_scores_weak,per_class_losses_weak]  = doEvaluation( opts,imdb,20,28,'weak');

%%
figure,plot(F_scores_orig(999,1:28),'g-');
hold on;plot(F_scores_weak(999,1:28),'r-');
legend('orig','weak');

figure,plot(mean(F_scores(:,1:28)),'g-d');
F_scores_weak(isnan(F_scores_weak) )= 0;
hold on;plot(mean(F_scores_weak(:,1:28)),'r-d');
% legend('orig','weak');


figure,plot(F_scores_orig(:,end))

weighting = 1-F_scores_orig(:,end);
save weighting.mat weighting



%%
cm1 = exp(-cm/10);
cm1 = cm1.*(1-eye(1000));
% cm1 = cm1/sum(cm1(:));
z = linkage(squareform(cm1),'average');
c = cluster(z,'maxclust',100);
n = hist(c,1:30);
[r,ir] = sort(n,'ascend');
for it = 1:length(r)
    t = ir(it);
    sel_ = c==t;
    imdb.classes.description(sel_)'
    fprintf('***************************************************\n')
    pause;
    clc
end
dendrogram(z)

[r,ir] = max(aucs,[],2);

figure,hist(r-aucs(:,end))
figure,hist(ir,1:20)

figure,plot(ir,r,'r+')

% f_scores = zeros(

t = 1;
plot(aucs(t,:))


size(aucs)
figure,plot(aucs')

[IDX,C] = kmeans2(aucs,25);

% who got clustered together?
%%imdb.classes.description(IDX==22)'
AA = aucs*aucs';

[v,iv] = sort(AA,2,'descend');
figure,imagesc(v)

for t = 1:1000
    t
    imdb.classes.description(iv(t,1:5))'
    dpc
end


% try to find correlations between classes
% aucs = aucs_bkp;
aucs = normalize_vec(aucs')';