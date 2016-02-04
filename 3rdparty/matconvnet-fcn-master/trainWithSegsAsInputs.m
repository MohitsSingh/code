function [imdb_aug,net,finalModelPath] = trainWithSegsAsInputs(class_names,baseDir, subsets,imdb_orig,sel_,suffix)

gpuDevice(2);

iSubset = 1;
f = find(subsets(iSubset,:));
subsetName = concat_names(class_names,f,['conditional_insert_preds_div100_' suffix '_']);
expDir = fullfile(baseDir,subsetName);
nEpochs=50;
finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);

baseModelPath = '/home/amirro/storage/matconv_data/pascal-fcn8s-dag.mat';
net = load(baseModelPath) ;
net = dagnn.DagNN.loadobj(net) ;
net.mode = 'test' ;
predVar = 'prediction';
inputVar = 'data' ;
net.move('gpu');


predsPath = fullfile('~/storage/misc',[subsetName '.mat']);
if exist(predsPath,'file')
    load(predsPath);
else
    
    preds = {};
    for u = 1:length(imdb_orig.images_data)
        u
        rgb = imdb_orig.images_data{u};
        [scores,pred] = applyNet(net,single(rgb),true,'data','upscore');
        softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
        % we want only a seleted output  prediction for now.
        if ~isempty(sel_)
            preds{u} =  softmaxScores(:,:,sel_);
        else
            preds{u} =  softmaxScores;
        end
    end
    save(predsPath,'preds','-v7.3');
end
% save ~/storage/misc/person_preds.mat person_preds

%%
% load  ~/storage/misc/person_preds.mat
%%

%%
% gpuDevice(2);

%%
%initialize  the network
opts.modelType = 'fcn8s' ;
opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
net = fcnInitializeModel_action_obj('sourceModelPath', opts.sourceModelPath,'nClasses',imdb_orig.nClasses) ;
net = fcnInitializeModel16s(net) ;
net = fcnInitializeModel8s(net) ;
%%%%%%%%%%%%%%%%%%%%%%%%
%

net.move('gpu');

% now, initialize the first layer's extra filter layer to be
% the same as the mean of the others.

extendModel = true;

if extendModel
    sz = size(net.params(1).value);
    sz(3) = sz(3)+size(preds{1},3);
    V = gpuArray(zeros(sz(1),sz(2),sz(3),sz(4),'single'));
    V(:,:,1:3,:) = net.params(1).value;
    ss = size(V(:,:,4:end,:));
    V(:,:,4:end,:) = repmat(mean(V(:,:,1:3,:),3)/100,[1 1 ss(3) 1]);
    V(:,:,4:end,:) = V(:,:,4:end,:)+randn(ss)*.001;
    net.params(1).value = V;
end

% now let's predict the action object with this!

%subsetName = [subsetName '1'];
fprintf('%s\n',subsetName);
nn = 0;
% mean_person = 0;

if (extendModel)
    z = zeros(1,1,size(V,3));
    z(4:end) = 128;
    z(1:3) = net.meta.normalization.averageImage(1:3);
    net.meta.normalization.averageImage = z;
    net.meta.normalization.rgbMean = z;
end

%%

if ~exist(finalModelPath,'file')
    imdb_aug = imdb_orig;
    for t = 1:length(imdb_orig.labels)
        curLabel = imdb_orig.labels{t};
        curLabel(~ismember(curLabel,f)) = 0;
        imdb_aug.labels{t} = curLabel;
    end
    if extendModel
        for t = 1:length(imdb_orig.images_data)
            imdb_aug.images_data{t} = cat(3,imdb_aug.images_data{t},im2uint8(preds{t}));
        end
    end
    %net.layers(1).inputs = {'input'};
    %net.vars(1).name= 'input';
    
    my_fcn_train(imdb_aug,subsetName,150,struct('gpus',2,'resetGPU',false),net);
end

expDir = fullfile(baseDir,subsetName);
% finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);
test_params.set = 'val';
test_params.labels = {'none','face','hand','obj'};
test_params.labels_to_block = [];
test_params.prefix = 'perfs_ap';
[perfs,diags] = test_net_perf(expDir,1:50,imdb_aug,imdb_aug.train,imdb_aug.val,imdb_aug.test,test_params);
figure,plot(diags(:,4));

