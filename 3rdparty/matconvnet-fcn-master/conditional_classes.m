% Training of classes dependent on some other classes,
% such that training one class become possible or easy only once its predecessor
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
    rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');
    run matconvnet/matlab/vl_setupnn ;
    addpath matconvnet/examples ;
    imdbPath = '/net/mraid11/export/data/amirro/fcn/data/conditional_new_face_and_hand_and_obj/imdb.mat';
    addpath(genpath('~/code/3rdparty/plotpub'));
   
    load(imdbPath);
    imdb.train = train;imdb.val = val;imdb.test = test;
end

class_names = {'face','hand','obj'};
class_inds = [1,2,3];
nClasses = length(class_names);
imdb_orig = imdb;
subsets = allSubsets(nClasses);
subsets = subsets(2:end,:);
baseDir = '/net/mraid11/export/data/amirro/fcn/data/';

test_params.labels = {'none','face','hand','obj'};
test_params.labels_to_block = [];
test_params.prefix = 'perfs_ap';
test_params.set = 'val';
%%
gpuDevice(1);
nEpochs = 150;
figure(100); clf;
for iSubset = 1:length(subsets);
    iSubset
    f = find(subsets(iSubset,:));
    subsetName = concat_names(class_names,f,'conditional_new_');
    %subsetName = [subsetName '1'];
    fprintf('%s\n',subsetName);
    expDir = fullfile(baseDir,subsetName);
    finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
    
    %     modelFamily = 'matconvnet' ;
    %     [net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(finalModelPath, modelFamily);
    %     assert(net.layers(end).block.size(4) == 4)
    %     continue
    
    if ~exist(finalModelPath,'file')
        cur_imdb = imdb_orig;
        for t = 1:length(imdb.labels)
            curLabel = imdb.labels{t};
            curLabel(~ismember(curLabel,f)) = 0;
            cur_imdb.labels{t} = curLabel;
        end
        my_fcn_train(cur_imdb,subsetName,nEpochs,struct('gpus',2,'freeze',[]));
    end
    % measure network performances at different epochs.
    
    [perfs,diags] = test_net_perf(expDir,1:150,imdb,train,val,test,test_params);
    %     break
    subplot(2,4,iSubset);
    plot(diags(:,[1 1+f]),'LineWidth',2);
    legend(cat(2,{'bg',class_names{f}}))
    title(subsetName(13:end),'interpreter','none');
end

%% do it with a 32-stride
gpuDevice(1);
nEpochs = 50;
figure(100); clf;
for iSubset = 1:length(subsets);
    iSubset
    f = find(subsets(iSubset,:));
    subsetName = concat_names(class_names,f,'conditional_32_');
    %subsetName = [subsetName '1'];
    fprintf('%s\n',subsetName);
    expDir = fullfile(baseDir,subsetName);
    finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
    
    %     modelFamily = 'matconvnet' ;
    %     [net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(finalModelPath, modelFamily);
    %     assert(net.layers(end).block.size(4) == 4)
    %     continue
    
    if ~exist(finalModelPath,'file')
        cur_imdb = imdb_orig;
        for t = 1:length(imdb.labels)
            curLabel = imdb.labels{t};
            curLabel(~ismember(curLabel,f)) = 0;
            cur_imdb.labels{t} = curLabel;
        end
        my_fcn_train(cur_imdb,subsetName,nEpochs,struct('gpus',1,'freeze',[],'modelType','fcn32s'));
    end
    % measure network performances at different epochs.
    
    [perfs,diags] = test_net_perf(expDir,1:150,imdb,train,val,test,test_params);
    %     break
    subplot(2,4,iSubset);
    plot(diags(:,[1 1+f]),'LineWidth',2);
    legend(cat(2,{'bg',class_names{f}}))
    title(subsetName(13:end),'interpreter','none');
end
%%
maximizeFigure;
im = export_fig;
imwrite(im,'~/notes/conditional/conditional_learning_corrected.png');

%% try seeing if training the object alone with a smaller learning rate helps:
f = find(subsets(1,:));
subsetName = concat_names(class_names,f,'conditional_lr_');
fprintf('%s\n',subsetName);
expDir = fullfile(baseDir,subsetName);
finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
if ~exist(finalModelPath,'file')
    cur_imdb = imdb_orig;
    for t = 1:length(imdb.labels)
        curLabel = imdb.labels{t};
        curLabel(~ismember(curLabel,f)) = 0;
        cur_imdb.labels{t} = curLabel;
    end
    my_fcn_train(cur_imdb,subsetName,nEpochs,struct('gpus',1,'lr_ratio',.1));
end
%
[perfs,diags] = test_net_perf(expDir,50,cur_imdb,train,val,test,test_params);
plot(diags(:,:),'LineWidth',2);legend('bg','face','hand','obj');


%% Use the hand+obj network as a base model for training only the object.
subsetName = concat_names(class_names,f,'conditional_new_');
expDir = fullfile(baseDir,subsetName);
modelPath = fullfile(expDir,['net-epoch-' num2str(50) '.mat']);

cur_imdb = imdb_orig;
f = find(subsets(6,:));
% subsetName = concat_names(class_names,f,'conditional_');

subsetName = [subsetName '_E_obj'];
[net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(modelPath, 'matconvnet');
% Add loss layer
net.addLayer('objective', ...
    SegmentationLoss('loss', 'softmaxlog'), ...
    {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
    SegmentationAccuracy(), ...
    {'prediction', 'label'}, 'accuracy') ;
nEpochs=50;
my_fcn_train(cur_imdb,subsetName,nEpochs,struct('gpus',1),net);

% check the performance of the resulting network.
expDir = fullfile(baseDir,subsetName);
test_params.set = 'val';
[perfs,diags] = test_net_perf(expDir,1:50,imdb,train,val,test,test_params);
figure(2); clf;
plot(diags(:,:),'LineWidth',2);legend('bg','face','hand','obj');
title(subsetName(13:end),'interpreter','none');

%% test the relations between the classes:
% block out the face and hand and check if the action object is still
% detected.
addpath('~/code/3rdparty/Inpaint_nans/');
f = find(subsets(7,:));
subsetName = concat_names(class_names,f,'conditional_');
fprintf('%s\n',subsetName);
expDir = fullfile(baseDir,subsetName);

% labels_to_keep = eye(3)>0;
% labels_to_block = ~labels_to_keep;
% labels_to_block = subsets;
%%
% figure(3); clf;
%
% for ii = 1:size(subsets,1)
%     ii
%     labels_to_block = find(subsets(ii,:));
%     perfName = concat_names(class_names,labels_to_block,'perf_blk_');
%     %     subplot(1,3,1);
%     test_params.labels_to_block = labels_to_block;
%     test_params =
%     [perfs,diags] = test_net_perf(expDir,50,imdb,train,val,test,test_params);
%     test_params.labels_to_block = [];
%     figure(3);
%     subplot(2,4,ii);
%     imagesc2(perfs.cm_n(3:4,3:4));%;axis equal;
%     title(perfName,'interpreter','none');
%     % dpc
% end

%% check if the observed effect is a function of the training set size:
% train on almost all images, *including the test set* an leave a bit for
% validation
f = find(subsets(1,:));
subsetName = concat_names(class_names,f,'conditional_new_more_data_');
fprintf('%s\n',subsetName);
expDir = fullfile(baseDir,subsetName);
finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
if ~exist(finalModelPath,'file')
    cur_imdb = imdb_orig;
    cur_imdb.train = [cur_imdb.train cur_imdb.test];
    cur_imdb.test = [];
    for t = 1:length(imdb.labels)
        curLabel = imdb.labels{t};
        curLabel(~ismember(curLabel,f)) = 0;
        cur_imdb.labels{t} = curLabel;
    end
    my_fcn_train(cur_imdb,subsetName,100,struct('gpus',1));
end

[perfs,diags] = test_net_perf(expDir,1:63,cur_imdb,train,val,test,test_params);
figure(2); clf;
plot(diags(:,:),'LineWidth',2);legend('bg','face','hand','obj');

% we can see that indeed using a much bigger training set eventually
% enables training, but the final performance is still lower compared to
% using the face and hands.

%%
% let's start with a pre-trained network on PASCAL, e.g, tuned to find 21
% object classes, and extend it to predict action objects as well,
% by adding the network's outputs as input to the learned network.

%[imdb_aug,net,finalModelPath] = trainWithSegsAsInputs(16);
% [cur_imdb,net,finalModelPath] = trainWithSegsAsInputs(class_names,baseDir, subsets,imdb_orig,16,'person');
addpath('~/code/3rdparty');
[cur_imdb,net,finalModelPath] = trainWithSegsAsInputs(class_names,baseDir, subsets,imdb_orig,[],'all');


% % % baseModelPath = '/home/amirro/storage/matconv_data/pascal-fcn8s-dag.mat';
% % % net = load(baseModelPath) ;
% % % net = dagnn.DagNN.loadobj(net) ;
% % % net.mode = 'test' ;
% % % predVar = 'prediction';
% % % inputVar = 'data' ;
% % % 
% % % net.move('gpu');
% % % person_preds = {};
% % % for u = 1:length(imdb.images_data)
% % %     u
% % %     rgb = imdb.images_data{u};
% % %     [scores,pred] = applyNet(net,single(rgb),true,'data','upscore');
% % %     softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
% % %     % we want only the 16'th output (person) prediction for now.
% % %     person_preds{u} = softmaxScores(:,:,16);
% % % end
% % % 
% % % save ~/storage/misc/person_preds.mat person_preds
% % % 
% % % %%
% % % load  ~/storage/misc/person_preds.mat
% % % %%
% % % 
% % % %%
% % % gpuDevice(1);
% % % 
% % % %%
% % % %initialize  the network
% % % opts.modelType = 'fcn8s' ;
% % % opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
% % % net = fcnInitializeModel_action_obj('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses) ;
% % % net = fcnInitializeModel16s(net) ;
% % % net = fcnInitializeModel8s(net) ;
% % % %%%%%%%%%%%%%%%%%%%%%%%%
% % % %
% % % 
% % % % baseModelPath = '/home/amirro/storage/matconv_data/pascal-fcn8s-dag.mat';
% % % % net = load(baseModelPath) ;
% % % % net = dagnn.DagNN.loadobj(net) ;
% % % % net.mode = 'test' ;
% % % % predVar = 'prediction';
% % % % inputVar = 'data' ;
% % % % 
% % % % net.renameVar('data','input');
% % % net.move('gpu');
% % % % 
% % % % net.addLayer('objective', ...
% % % %     SegmentationLoss('loss', 'softmaxlog'), ...
% % % %     {'upscore', 'label'}, 'objective') ;
% % % % 
% % % % % Add accuracy layer
% % % % net.addLayer('accuracy', ...
% % % %     SegmentationAccuracy(), ...
% % % %     {'upscore', 'label'}, 'accuracy') ;
% % % 
% % % % now, initialize the first layer's extra filter layer to be
% % % % the same as the mean of the others.
% % % 
% % % extendModel = true;
% % % 
% % % if extendModel
% % %     sz = size(net.params(1).value);
% % %     sz(3) = sz(3)+1;
% % %     V = gpuArray(zeros(sz(1),sz(2),sz(3),sz(4),'single'));
% % %     V(:,:,1:3,:) = net.params(1).value;
% % %     V(:,:,4,:) = mean(V(:,:,1:3,:),3)/100;
% % %     net.params(1).value = V;
% % % end
% % % 
% % % % now let's predict the action object with this!
% % % 
% % % iSubset=1
% % % f = find(subsets(iSubset,:));
% % % subsetName = concat_names(class_names,f,'conditional_insert_person_div100_');
% % % %subsetName = [subsetName '1'];
% % % fprintf('%s\n',subsetName);
% % % expDir = fullfile(baseDir,subsetName);
% % % nEpochs=150
% % % finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
% % % nn = 0;
% % % % mean_person = 0;
% % % 
% % % if (extendModel)
% % % z = zeros(1,1,4);
% % % z(4) = 128;
% % % z(1:3) = net.meta.normalization.averageImage(1:3);
% % % net.meta.normalization.averageImage = z;
% % % net.meta.normalization.rgbMean = z;
% % % end
% % % 
% % % %%
% % % 
% % % if ~exist(finalModelPath,'file')
% % %     cur_imdb = imdb_orig;
% % %     for t = 1:length(imdb.labels)
% % %         curLabel = imdb.labels{t};
% % %         curLabel(~ismember(curLabel,f)) = 0;
% % %         cur_imdb.labels{t} = curLabel;
% % %     end
% % %     if extendModel
% % %         for t = 1:length(imdb.images_data)
% % %             cur_imdb.images_data{t} = cat(3,cur_imdb.images_data{t},im2uint8(person_preds{t}));
% % %         end
% % %     end
% % %     %net.layers(1).inputs = {'input'};
% % %     %net.vars(1).name= 'input';        
% % %    
% % %     my_fcn_train(cur_imdb,subsetName,150,struct('gpus',1,'resetGPU',false),net);
% % % end

%% test the net a bit...
expDir = fullfile(baseDir,subsetName);
% finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
test_params.set = 'val';
[perfs,diags] = test_net_perf(expDir,1:150,cur_imdb,train,val,test,test_params);
figure,plot(diags(:,4));

%% The resulant accuracy is nice, but I doubt if this had to do with the person prediction input layer.
%% The final accuracy after ~150 epochs reaches around .36. Compare with the original network
% trained on the action object alone : it reaches almost the same, .354,
% and starts learning faster. 

%% now try starting with a network pre-trained for semantic segmentation:
baseModelPath = '/home/amirro/storage/matconv_data/pascal-fcn8s-dag.mat';
net = load(baseModelPath) ;
net = dagnn.DagNN.loadobj(net) ;
% net.mode = 'test' ;
predVar = 'prediction';
inputVar = 'data' ;

net.addLayer('objective', ...
    SegmentationLoss('loss', 'softmaxlog'), ...
    {'upscore', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
    SegmentationAccuracy(), ...
    {'upscore', 'label'}, 'accuracy') ;

for u = 1:length(net.params)
    sz = size(net.params(u).value);    
    sz(sz==21) = 4;
    if length(sz)==4
        net.params(u).value = net.params(u).value(1:sz(1),1:sz(2),1:sz(3),1:sz(4));
    elseif length(sz) == 3
        net.params(u).value = net.params(u).value(1:sz(1),1:sz(2),1:sz(3));
    elseif length(sz) == 2
        net.params(u).value = net.params(u).value(1:sz(1),1:sz(2));
    end        
end
%% let's try to change the learning scheme to update only weights relating to to target object.

iSubset = 1;
f = find(subsets(iSubset,:));
subsetName = concat_names(class_names,f,['conditional_obj_from_fcn8_pas_dag_']);
expDir = fullfile(baseDir,subsetName);
nEpochs=150;
finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);


cur_imdb = imdb_orig;
for t = 1:length(imdb.labels)
    curLabel = imdb.labels{t};
    curLabel(~ismember(curLabel,f)) = 0;
    cur_imdb.labels{t} = curLabel;
end

net.renameVar('data','input');

my_fcn_train(cur_imdb,subsetName,nEpochs,struct('gpus',2,'freeze',[]),net);

% check the performance of the resulting network.
[perfs,diags] = test_net_perf(expDir,nEpochs:-1:1,imdb,train,val,test,test_params);
figure(2); clf;
plot(diags(:,:),'LineWidth',2);legend('bg','face','hand','obj');
title(subsetName(13:end),'interpreter','none');
%%

% show the net topology.

all_edges = {};
for t = 1:length(net.layers)
    ins = net.layers(t).inputIndexes;
    outs = net.layers(t).outputIndexes;
    
    if length(ins)<length(outs)
        ins = repmat(ins,1,length(outs));
    end
    if (length(ins)>length(outs))
        outs = repmat(outs,1,length(ins));
    end
    all_edges{end+1} = [ins' outs'];
    
    %     in_names = net.layers(t).inputs;
    %     out_names = net.layers(t).outputs;
    
end
all_edges = cat(1,all_edges{:});
names = {net.layers.name,'prediction'};

% xy = rand(length(names),2);

for p = 1:size(all_edges,1)
    
end

% all_edges = all_edges(1:end-1,:);
A = sparse(all_edges(:,1),all_edges(:,2),1);



gplot(A,xy)


%%

tarnames = textread('/home/amirro/tars.txt','%s\n')
jpeg_files = textread('/home/amirro/files.txt','%s\n')

tarnames = cellfun2(@(x) x(1:9),tarnames);
jpgnames = cellfun2(@(x) x(1:9),jpeg_files);

[c,ia,ib] = intersect(tarnames,jpgnames);
