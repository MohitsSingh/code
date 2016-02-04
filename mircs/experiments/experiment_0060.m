%% 16/2/2015
% deep bag-of-words classification
% extract a bag-of-features style representation but use deep network
% features instead of sift for features.
%
% initpath;
% config;
%
% start on, e.g, caltech or something...

% addpath('/home/amirro/storage/data/caltech101/101_ObjectCategories');
% addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');

%%
function phow_caltech101()

conf.calDir = '/home/amirro/storage/data/caltech101/101_ObjectCategories/' ;
conf.dataDir = 'data/' ;
conf.autoDownloadData = true ;
conf.numTrain = 30 ;
conf.numTest = 30 ;
conf.numClasses = 102 ;
conf.foolTheSystem = true;
conf.numWords = 600 ;
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 10 ;
% load('data/cifar-baseline/net-epoch-20.mat');
% load /home/amirro/code/3rdparty/matconvnet-1.0-beta7/examples/cifar_data_mean.mat; % data_mean
% net.layers = net.layers(1:10);

networkPath = 'imagenet-vgg-s.mat';
net = load(networkPath);
net.layers = net.layers(1:15);
%
% net.data_mean = data_mean;
conf.svm.solver = 'sdca' ;
%conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
conf.phowOpts = {'Step', 3} ;
conf.clobber = false ;
conf.tinyProblem = false ;
conf.prefix = 'baseline' ;
conf.randSeed = 1 ;

if conf.tinyProblem
    conf.prefix = 'tiny-8' ;
    conf.numClasses = 5 ;
    conf.numSpatialX = [1 2]
    conf.numSpatialY = [1 2];
    conf.numWords = 300 ;
    conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5} ;
end

conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

% --------------------------------------------------------------------
%                                            Download Caltech-101 data
% --------------------------------------------------------------------

if ~exist(conf.calDir, 'dir') || ...
        (~exist(fullfile(conf.calDir, 'airplanes'),'dir') && ...
        ~exist(fullfile(conf.calDir, '101_ObjectCategories', 'airplanes')))
    if ~conf.autoDownloadData
        error(...
            ['Caltech-101 data not found. ' ...
            'Set conf.autoDownloadData=true to download the required data.']) ;
    end
    vl_xmkdir(conf.calDir) ;
    calUrl = ['http://www.vision.caltech.edu/Image_Datasets/' ...
        'Caltech101/101_ObjectCategories.tar.gz'] ;
    fprintf('Downloading Caltech-101 data to ''%s''. This will take a while.', conf.calDir) ;
    untar(calUrl, conf.calDir) ;
end

if ~exist(fullfile(conf.calDir, 'airplanes'),'dir')
    conf.calDir = fullfile(conf.calDir, '101_ObjectCategories') ;
end

% --------------------------------------------------------------------
%                                                           Setup data
% --------------------------------------------------------------------
classes = dir(conf.calDir) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;

images = {} ;
imageClass = {} ;
for ci = 1:length(classes)
    ims = dir(fullfile(conf.calDir, classes{ci}, '*.jpg'))' ;
    ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
    ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
    images = {images{:}, ims{:}} ;
    imageClass{end+1} = ci * ones(1,length(ims)) ;
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain) ;
selTest = setdiff(1:length(images), selTrain) ;
imageClass = cat(2, imageClass{:}) ;

model.classes = classes ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;

% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------



if ~conf.foolTheSystem && (~exist(conf.vocabPath) || conf.clobber)
    
    % Get some PHOW descriptors to train the dictionary
    selTrainFeats = vl_colsubset(selTrain, 30) ;
    descrs = {} ;
    for ii = 1:length(selTrainFeats)
        ii
        %   parfor ii = 1:length(selTrainFeats)
        im = imread(fullfile(conf.calDir, images{selTrainFeats(ii)})) ;
%         clf; imagesc2(im); pause;
        im = standarizeImage(im) ;
        %[drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
        [drop, descrs{ii}] = dense_cnn_descs(im,net,conf);
    end
    
    descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
    descrs = single(descrs) ;
    
    % Quantize the descriptors to get the visual words
    vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
    save(conf.vocabPath, 'vocab') ;
else
    vocab = [];
    if ~conf.foolTheSystem
        load(conf.vocabPath) ;
    end
    
end

model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
    if (any(vocab))
        model.kdtree = vl_kdtreebuild(vocab) ;
    end
end

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------

% 
if (conf.foolTheSystem)
    %%MM = cellfun2(@(x) imread2(x),fullfile(conf.calDir,images));
%     MM = vl_imreadjpeg(fullfile(conf.calDir,images));
    MM = cellfun2(@(x) imread2(x),fullfile(conf.calDir,images));
    tiles = [1 1;2 2;3 3];
%     tiles = [1 1];
    [res] = extractDNNFeats_tiled(MM,net,tiles,[16],false);
    save  ~/storage/misc/res.mat res -v7.3
    x_orig = res.x;
     x =  reshape(x_orig,[],size(x_orig,2)/sum(prod(tiles,2),1));
end

if ~exist(conf.histPath) || conf.clobber
    hists = {} ;
    for ii = 1:length(images)
        ii
        % for ii = 1:length(images)
        fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
        im = imread(fullfile(conf.calDir, images{ii})) ;
        hists{ii} = getImageDescriptor(model, im,net,conf);
    end
    
    hists = cat(2, hists{:}) ;
    save(conf.histPath, 'hists') ;
else
    load(conf.histPath) ;
end

% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------
%%
hists = x;%res(1).x(1:1:end,:);
% [~,mu,stds] = normalizeData(hists(:,selTrain));
% hists = normalizeData(hists,mu,stds);
% hists = normalizeData(hists);
% hists = normalize_vec(hists);
% psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;
psix = hists(1:10:end,:);%:end,:);%4096,:);
% psix = hists(1:end,:);
psix = hists(1:3*4096,:);

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------
conf.svm.C = .0001;
% close all=
% lambda = .0001;
% conf.svm.solver = 'sdca';
conf.svm.solver = 'liblinear';
if ~exist(conf.modelPath) || true%conf.clobber
    switch conf.svm.solver
        case {'sgd', 'sdca'}
            lambda = 1 / (conf.svm.C *  length(selTrain)) ;
            w = [] ;
            for ci = 1:length(classes)
                ci
                perm = randperm(length(selTrain)) ;
                fprintf('Training model for class %s\n', classes{ci}) ;
                y = 2 * (imageClass(selTrain) == ci) - 1 ;
                %         [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
                %           'Solver', conf.svm.solver, ...
                %           'MaxNumIterations', 50/lambda, ...
                %           'BiasMultiplier', conf.svm.biasMultiplier, ...
                %           'Epsilon', 1e-3);
                [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
                    'Solver', conf.svm.solver, ...                    
                    'BiasMultiplier', conf.svm.biasMultiplier,'MaxNumIterations', 50);
            end
            
        case 'liblinear'
            svm = train(imageClass(selTrain)', ...
                sparse(double(psix(:,selTrain))),  ...
                sprintf(' -s 4 -B %f -c %f', ...
                conf.svm.biasMultiplier, conf.svm.C), ...
                'col') ;
            w = svm.w(:,1:end-1)' ;
            b =  svm.w(:,end)' ;
    end
    
    model.b = conf.svm.biasMultiplier * b ;
    model.w = w ;
    
    save(conf.modelPath, 'model') ;
else
    load(conf.modelPath) ;
end

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------

% Estimate the class of the test images
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;

% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
    imageClass(selTest), imageEstClass(selTest)) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
figure(1) ; clf;
subplot(1,2,1) ;
imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,2,2) ;
imagesc(confus) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
    100 * mean(diag(confus)/conf.numTest) )) ;
%%
return;
print('-depsc2', [conf.resultPath '.ps']) ;
save([conf.resultPath '.mat'], 'confus', 'conf') ;

% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im; 
%im = im2single(im) ;
%if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end

% -------------------------------------------------------------------------
function hist = getImageDescriptor(model, im,net,conf)
% -------------------------------------------------------------------------

im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
%[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;
[frames, descrs] = dense_cnn_descs(im, net,conf);
if conf.foolTheSystem
    hist = descrs;
    return
end

% quantize local descriptors into visual words
switch model.quantizer
    case 'vq'
        [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
    case 'kdtree'
        binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
            single(descrs), ...
            'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
    binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
    binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;
    
    % combined quantization
    bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
        binsy,binsx,binsa) ;
    hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
    hist = vl_binsum(hist, ones(size(bins)), bins) ;
    hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;

% -------------------------------------------------------------------------
function [className, score] = classify(model, im)
% -------------------------------------------------------------------------

hist = getImageDescriptor(model, im,net) ;
psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes{best} ;

function [frames,descs] = dense_cnn_descs(im,net,conf)
% first model: remove data mean from each sub-image independently.
% second model: remove 128 arbitrarily from all pixels for better
% speed.
if (length(size(im))==2)
    im = cat(3,im,im,im);
end

%     mm = 17;
%     im = single(im);
%     loc_min = [mm mm];
%     loc_max = fliplr(size2(im))-mm;
%     D = 10;
%     [xx,yy] = meshgrid(loc_min(1):D:loc_max(1),loc_min(2):D:loc_max(2));
%     xx = xx(:); yy = yy(:);
%     frames = [xx yy]';
%     descs = [];

if (conf.foolTheSystem)
    frames = [10;10];
    U = vl_simplenn(net, single(imResample(im,[224 224],'bilinear')));    
    descs = col(U(16).x);
else
%     im = single(im2uint8(im));
    
%     U = extractDNNFeats_tiled(im,net,[3 3],16,false);     
    %descs = U(11).x;
    descs = U.x;
    sz = size2(descs);
    [xx,yy] = meshgrid(1:size(descs,2),1:size(descs,1));
    yy = (yy/sz(1))*(size(im,1)-1); % this is to avoid elements on the image border
    xx = (xx/sz(2))*(size(im,2)-1);
    %     x2(uint8(im)); plot(xx(:),yy(:),'r.');
    xx = xx(:); yy = yy(:);
    frames = [xx yy]';
    descs = reshape(descs,[],64)';
end


% approximate the location by just "blowing up" the original
% coordinates
%     rects = inflatebbox([xx yy xx yy],[32 32],'both',true);
%     sub_imgs = multiCrop2(im,rects);
%     sub_imgs = cat(4,sub_imgs{:});
%     sub_imgs = bsxfun(@minus,sub_imgs,net.data_mean);

%dnn_res = vl_simplenn(net, sub_imgs(:,:,:,:));
%     size(dnn_res(11).x);
%     imo = prepareForDNN(batch,net);
%     dnn_res = vl_simplenn(net, imo);
%     [res] = extractDNNFeats(sub_imgs,net);


%[drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;