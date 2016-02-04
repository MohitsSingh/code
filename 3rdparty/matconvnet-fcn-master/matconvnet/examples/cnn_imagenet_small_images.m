function cnn_imagenet_small_images(varargin)
% CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%   This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%   VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

if (0)
    addpath(genpath('~/code/utils'));
    addpath(genpath('~/code/3rdparty/piotr_toolbox'));
    addpath('~/code/3rdparty/sc');
    addpath('~/code/3rdparty/export_fig');
    addpath('utils/');
    addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
    vl_setup
    rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');
    run(fullfile(fileparts(mfilename('fullpath')), ...
        '..', 'matlab', 'vl_setupnn.m')) ;
end
%%
dataDir_orig = '/home/amirro/storage/data/ILSVRC2012/';
dataDir_small = '/home/amirro/storage/data/ILSVRC2012_small/';

% train images
cats = dir(fullfile(dataDir_orig,'images','train'));
cats  = cats(3:end);
cats = cats([cats.isdir]);
cats = {cats.name};
my_color_tform = makecform('cmyk2srgb');
% % for iCat = 1:length(cats)
% %     iCat
% %     in_dir = fullfile(dataDir_orig,'images','train',cats{iCat});
% %     out_dir = fullfile(dataDir_small,'images','train',cats{iCat});
% %     ensuredir(out_dir);
% %     dd = getAllFiles(in_dir,'*.JPEG');
% %     my_imread_and_shrink_set_of_images(dd,in_dir,out_dir,128,my_color_tform);
% %     %     for ii = 1:length(dd)
% %     %         if mod(ii,50)==0
% %     %             ii
% %     %         end
% %     %         outPath = strrep(dd{ii},in_dir,out_dir);
% %     %         if exist(outPath,'file'),continue,end
% %     %         I = my_imread_and_shrink(dd(ii),128,my_color_tform);
% %     %         imwrite(I,outPath);
% %     %     end
% % end
% same for val
% in_dir = fullfile(dataDir_orig,'images','val');
% out_dir = fullfile(dataDir_small,'images','val');
% ensuredir(out_dir);
% dd = getAllFiles(in_dir,'*.JPEG');
% my_imread_and_shrink_set_of_images(dd,in_dir,out_dir,128,my_color_tform);

% same for test
in_dir = fullfile(dataDir_orig,'images','test');
out_dir = fullfile(dataDir_small,'images','test');
ensuredir(out_dir);
%dd = getAllFiles(in_dir,'*.JPEG');
% dd = readlines('/home/amirro/big_test.txt');
%dir(fullfile(in_dir,'*.JPEG'));

% my_imread_and_shrink_set_of_images(dd,in_dir,out_dir,128,my_color_tform);

% for ii = 1:length(dd)
%     if mod(ii,50)==0
%         ii
%     end
%     outPath = strrep(dd{ii},in_dir,out_dir);
%     if exist(outPath,'file'),continue,end
%     I = my_imread_and_shrink(dd(ii),128,my_color_tform);
%     imwrite(I,outPath);
% end

%%

opts.dataDir = '/home/amirro/storage/data/ILSVRC2012_small/';
opts.modelType = 'vgg-f' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;

[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
% opts.expDir = fullfile('data', sprintf('imagenet12-%s-%s', ...
%                                        sfx, opts.networkType)) ;

sfx = 'vgg-f-bnorm-resize';

opts.expDir = fullfile('/home/amirro/storage/data/', sprintf('imagenet12-small-%s-%s', ...
    sfx, opts.networkType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 24 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = 2;
opts.train.prefetch = true ;
opts.train.sync = true ;
opts.train.cudnn = false ;
opts.train.expDir = opts.expDir ;

if ~opts.batchNormalization
    opts.train.learningRate = logspace(-2, -4, 60) ;
else
    opts.train.learningRate = logspace(-1, -4, 20) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

% opts.imdbPath ='/home/amirro/storage/data/imagenet12-small-vgg-f-bnorm-simplenn/imdb_small.mat';

opts.imdbPath = '/home/amirro/storage/data/imagenet12-small-vgg-f-bnorm-simplenn/imdb.mat';
if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
else
    imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% % sel_ = vl_colsubset(1:length(imdb.images.set),5000,'Uniform');
% % imdb.images.set = imdb.images.set(sel_);
% % imdb.images.label = imdb.images.label(sel_);
% % imdb.images.name = imdb.images.name(sel_);
% % imdb.images.id = imdb.images.id(sel_);


% imdb.images.data = vl_imreadjpeg(fullfile(imdb.imageDir,imdb.images.name));
% save(opts.imdbPath,'-struct','imdb');
% for t = 1:length(imdb.images.data)
%     imdb.images.data{t} = uint8(imdb.images.data{t})
% end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = cnn_imagenet_init('model', opts.modelType, ...
    'batchNormalization', opts.batchNormalization, ...
    'weightInitMethod', opts.weightInitMethod) ;
%net.normalization.imageSize = [128 128 3];
bopts = net.normalization ;
bopts.numThreads = opts.numFetchThreads ;

% compute image statistics (mean, RGB covariances etc)
%imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
imageStatsPath = '/home/amirro/storage/data/imagenet12-small-vgg-f-bnorm-simplenn/imageStats.mat';

if exist(imageStatsPath)
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
    [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
    save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% One can use the average RGB value, or use a different average for
% each pixel
%net.normalization.averageImage = averageImage ;
net.normalization.averageImage = rgbMean ;

switch lower(opts.networkType)
    case 'simplenn'
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'top1error') ;
    otherwise
        error('Unknown netowrk type ''%s''.', opts.networkType) ;
end

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

[v,d] = eig(rgbCovariance) ;
bopts.transformation = 'stretch' ;
bopts.averageImage = rgbMean ;
bopts.rgbVariance = 0.1*sqrt(d)*v' ;
useGpu = numel(opts.train.gpus) > 0 ;

switch lower(opts.networkType)
    case 'simplenn'
        fn = getBatchSimpleNNWrapper(bopts) ;
        [net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true) ;
    case 'dagnn'
        fn = getBatchDagNNWrapper(bopts, useGpu) ;
        opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
        info = cnn_train_dag(net, imdb, fn, opts.train) ;
end

sel_test = find(imdb.images.set == 2); % val

fileNames = fullfile(imdb.imageDir,imdb.images.name(sel_test));
[test_labels,iv] = sort(imdb.images.label(sel_test));
test_labels = test_labels(1:1:end);
fileNames = fileNames(iv(1:1:end));
batches = batchify(length(fileNames),length(fileNames)/256);
% profile on
test_images = {};
for z = 1:length(batches)
    z
    curBatch = batches{z};
    q = vl_imreadjpeg(fileNames(curBatch),'NumThreads',12);
    if z < length(batches)
        nextBatch = batches{z+1};
        vl_imreadjpeg(fileNames(nextBatch),'NumThreads',12,'Prefetch');
    end
%     for t = 1:length(q)
%         q{t} = uint8(q{t});
%     end
    batch_images=q;
    test_images{z} = cnn_imagenet_get_batch(batch_images,bopts);   
end
% profile viewer
for t = 1:length(batches)
    t     
       
end
all_results = {};
all_cms = {};
perfDir = '/home/amirro/storage/data/imagenet12-small-vgg-f-bnorm-simplenn/perf/';
for ii = 3:20
    ii
    
    curOutPath = fullfile(perfDir,['epoch_' num2str(ii) '.mat']);
    if exist(curOutPath,'file')
        load(curOutPath)
    else
        gpuDevice(2);
        load(sprintf('%s/net-epoch-%d.mat',opts.expDir,ii))
        net = vl_simplenn_move(net, 'gpu') ;
        net.layers = net.layers(1:end-1);
        %     test_images = test_images;
        %     test_labels = test_labels(1:500);
        
        [cm,curResults] = calcPerf2(net,test_images,test_labels,1);
        %     test_images = imdb.images.data(:,:,:,imdb.images.set==3);
        %     test_labels = imdb.images.labels(imdb.images.set==3);
        save(curOutPath,'cm','curResults');
    end
    all_cms{ii} = cm;
    all_results{ii} = curResults;
end


aucs = zeros(1000,length(all_cms));
for t = 1:length(all_results)
    t
    t
    curResults = all_results{t};
    E = exp(bsxfun(@minus, curResults, max(curResults,[],1))) ;
    L = sum(E,1) ;
    curResults = bsxfun(@rdivide, E, L) ;
    for tt = 1:1000
        [prec,rec,info] = vl_pr(2*(test_labels==tt)-1,curResults(tt,:));
        aucs(tt,t) = info.ap;
    end    
end
F_scores = zeros(1000,20);
for t = 1:length(all_results)
    t
    curResults = all_results{t};
    [~,IDXpred] = max(curResults,[],1);
    cm = confMatrix( test_labels, IDXpred, 1000 );
    recall = diag(bsxfun(@rdivide,cm,sum(cm,2)));
    precision = diag(bsxfun(@rdivide,cm,sum(cm,1)));
    F_scores(:,t) = (2*precision.*recall)./(precision+recall);
end

save /home/amirro/storage/data/imagenet12-small-vgg-f-bnorm-simplenn/perf/aucs_and_scores.mat F_scores aucs

F_scores2 = F_scores;
F_scores2(isnan(F_scores2)) = 0;




figure,plot(mean(F_scores2));

% aucs_bkp = aucs;
aucs = F_scores2;
aucs_n = bsxfun(@rdivide,aucs,max(aucs,[],2));
aucs_n(isnan(aucs_n)) = 0;
%figure,plot(mean(aucs_n))
% find out the early and late bloomers
areas = zeros(1000,1);
for t = 1:1000
    p = 1:20;
    r = aucs_n(t,:);
    areas(t)=trapz(p,r);    
end

[v,iv] = sort(areas,'ascend');
for it = 1:10:length(iv)
    if (v(it)==0)
        continue
    end
    t = iv(it)
    y = aucs_n(t,:);
    clf;
    plot(y); title(sprintf('class:%s , max score: %f ', imdb.classes.description{t}, max(aucs(t,:))));
    hold on; plot(aucs(t,:));
    legend('normalized','original');
    dpc
end

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

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN2(imdb, batch, opts)
% -------------------------------------------------------------------------
images = imdb.images.data(batch);
im = cnn_imagenet_get_batch(images, opts, ...
    'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;

function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
    'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
    'prefetch', nargout == 0) ;
if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    inputs = {'input', im, 'label', imdb.images.label(batch)} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
    z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
    n = size(z,2) ;
    avg{t} = mean(temp, 4) ;
    rgbm1{t} = sum(z,2)/n ;
    rgbm2{t} = z*z'/n ;
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;


function my_imread_and_shrink_set_of_images(dd,in_dir,out_dir,newScale,my_color_tform)
%dd_out = getAllFiles(out_dir,'*.JPEG');
dd_out = readlines('~/small_test.txt');
% dd_finished = cellfun2(@(x) strrep(x,out_dir,in_dir),dd_out);
% if ~isempty(dd_finished)
alreadyFinishedFiles = ismember(dd,dd_out);
%     if all(alreadyFinishedFiles)
%         return
%     end
dd = dd(~alreadyFinishedFiles);
% end

dd_out = fullfile(out_dir,dd);
dd = fullfile(in_dir,dd);
%dd_out = cellfun2(@(x) strrep(x,in_dir,out_dir),dd);

% dd_in_batches = {};
% dd_out_batches = {};
batches = batchify(length(dd),round(length(dd)/200));
for iBatch = 1:length(batches)
    iBatch
    curBatch = batches{iBatch};
    cur_in = dd(curBatch);
    cur_out = dd_out(curBatch);
    II = vl_imreadjpeg(cur_in,'NumThreads',12);
    if iBatch < length(batches) % prepare the next batch already
        next_in = dd(batches{iBatch+1});
        vl_imreadjpeg(next_in,'NumThreads',12,'Prefetch','Preallocate',false);
    end
    for t = 1:length(II)
        I = II{t};
        if size(I,3)==4
            I(isnan(I)) = 0;
            I = applycform(double(I),my_color_tform);
        end
        I = I/255;
        sz = size2(I);
        f = max(newScale./sz);
        I = imResample(I,f,'bilinear');
        imwrite(I,cur_out{t});
    end
end



function I = my_imread_and_shrink(imPath,newScale,my_color_tform)
I = vl_imreadjpeg(imPath); I = I{1};
if size(I,3)==4
    I(isnan(I)) = 0;
    I = applycform(double(I),my_color_tform);
end

I = I/255;
sz = size2(I);
f = max(newScale./sz);
I = imResample(I,f,'bilinear');