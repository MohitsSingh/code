%% experiment 0043 -
%% 16/6/2014
%% desicion process to reach correct location : a patch may point to another location and scale
%% to continue the process. The "winning" value is the correct location in the image: location and scale.

addpath('/home/amirro/code/3rdparty/smallcode/');
%
% X = rand(10000,800);
% [recall, precision] = test(X, 256, 'ITQ');

default_init;
specific_classes_init;
% for iClass = 1:length(classes)
iClass
%%
%     cls = iClass;
iClass = 1;cls = iClass;
sel_train = class_labels ==cls & isTrain;
% parameters
% resPath = fullfile('~/storage/misc',[classNames{iClass} '_params.mat']);
% load (resPath);
% end

% ress = zeros(size(optParams));
% for t= 1:length(optParams);
%     ress(t) = sum(optParams(t).res(:,1)./optParams(t).res(:,2));
% end
% [v,iv] = max(ress);
% curParams=optParams(iv);
curParams = theParams;
curParams.img_h = 100;
% curParams.img_h = 100;
% curParams.nn = 100;
curParams.max_nn_checks = 0;
curParams.sample_max = false;
curParams.wSize = [4 4];
% curParams.nIter = 20;
sel_test = class_labels ==cls & ~isTrain;
curParams.normalizeWithFace = false;
[XX,offsets,all_scales,imgInds,subInds,values,kdtree,all_boxes,imgs] = ...
    preparePredictionData(conf,newImageData(validIndices(sel_train)),curParams);
f_test = validIndices(sel_test);
debug_ = true;
close all;

debugParams.debug = true;
debugParams.doVideo = false;
debugParams.showFreq =10;
debugParams.pause = .1;
curParams.nn = 50;
curParams.max_nn_checks = 0;curParams.nn;

%%
curParams.voteAll = false;

curParams.nIter = 50;
curParams.useSaliency = true;
for t =1:length(f_test)
    %     close all
    %     t=6
    curParams.min_scale = 1;
    curParams.sample_max = false;
%     t=15
    pMap = ...
        predictBoxes(conf,newImageData(f_test(t)),XX,curParams,offsets,all_scales,imgInds,subInds,values,imgs,all_boxes,kdtree,debugParams);
end
%%

