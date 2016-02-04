function [ rois,curScores,feats ] = detect_coarse(conf,imgData,params_coarse,featureExtractor,w_int,b_int)
params_coarse.sampling.boxSize = 1;
params_coarse.sampling.nBoxThetas = 25;
I = getImage(conf,imgData);
[rois,curScores,thetas,feats] = scoreCoarseRois(conf,imgData,params_coarse,featureExtractor,w_int,b_int);
% % [r,ir] = sort(curScores,'descend');
% % S = computeHeatMap(I,[rois ,normalise(curScores(:)).^2],'max');
% % 
% % figure(1); clf;
% % subplot(1,2,1); imagesc2(I);
% % subplot(1,2,2);
% % RR = sc(cat(3,S,im2double(I)),'prob');
% % imagesc2(RR);
% % title(num2str(max(curScores)));
% % 
