function  showPerformanceAnalysis( perf,nEpochs,F_scores )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


clf; 
subplot(2,2,1);plot(1:nEpochs,perf); title('performance vs epoch');
% find for each class it's maximal F_score.
[m,im] = max(F_scores,[],2);
subplot(2,2,2);hist(im,1:nEpochs); title('Distribution of best epochs');
% subplot(2,2,2); plot(im,m,'r+');
subplot(2,2,3);
boxplot(m,im); title('best epoch number vs. final performance');

