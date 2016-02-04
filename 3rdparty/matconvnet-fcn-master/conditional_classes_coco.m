% Training of classes dependent on some other classes,
% such that training one class become possible or easy only its predecessor
% has been trained.

% in this example, we are looking for action-objects, and we shall show
% that they are hard to detect when not given their predecessor classes.
% we start with an example of smoking people, where the possible image
% labels are background, hand, face and cigarette .

% we shall try the following orders of training:
% 1. train each class independently.
% 2. train all classes jointly.
% 3. train classes in some order: first,

% first, load the imdb.
if (0)
    addpath(genpath('~/code/utils'));
    addpath(genpath('~/code/3rdparty/piotr_toolbox'));
    addpath('~/code/3rdparty/sc');
    addpath('~/code/3rdparty/export_fig');
    addpath('utils/');
    addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
    vl_setup
    run matconvnet/matlab/vl_setupnn ;
    addpath matconvnet/examples ;
    
    addpath(genpath('~/code/3rdparty/plotpub'));
    rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');
    
    % make a dataset from pascal context
    imdbPath = '/net/mraid11/export/data/amirro/fcn/data/pas_context/imdb.mat';
    %%
    d = dir('/home/amirro/storage/data/pas_context/trainval');
%     z = '/home/amirro/storage/data/pas_context/VOCdevkit/    
    cd /net/mraid11/export/data/amirro/data/pas_context
    read_anno;
    cd /home/amirro/code/3rdparty/matconvnet-fcn-master
end

baseDir = '/net/mraid11/export/data/amirro/fcn/data/';
expDir = fullfile(baseDir,'coco_seg');
ensuredir(expDir);
%
%%
nEpochs = 20;
finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
if ~exist(finalModelPath,'file')
    my_fcn_train(imdb,'coco_seg',nEpochs,struct('gpus',1,'freeze',[]));
end
% measure network performances at different epochs.

test_params.labels = all_labels;
test_params.labels_to_block = [];
test_params.prefix = 'perfs_frac_10';
test_params.set = 'val';
train = imdb.train;val = imdb.val;test = imdb.test;
[perfs,diags] = test_net_perf(expDir,1:32,imdb,train,val(1:10:end),test,test_params);
%     break


FF = cat(2,perfs.F_score );
FF(isnan(FF)) = 0;
FF_normalized = bsxfun(@rdivide,FF,max(FF,[],2));
plot(FF_normalized')

FF_max = max(FF,[],2);
[v,iv] = sort(FF_max,'descend');

figure,imagesc(perfs(end).cm_n)


plot(diags)
figure(5)
legend(cat(2,{'bg',class_names{f}}))
title(subsetName(13:end),'interpreter','none');
diags
%%
maximizeFigure;
im = export_fig;
imwrite(im,'~/notes/conditional/conditional_learning_corrected.png');

