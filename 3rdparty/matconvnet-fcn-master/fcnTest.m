% function fcnTest(varargin)

run matconvnet/matlab/vl_setupnn;
addpath matconvnet/examples;
% experiment and data paths
opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-voc11' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/voc11' ;
opts.modelType = 'fcn32s' ;
% opts.expDir = 'data/fcn32s-voc11' ;
% opts.dataDir = 'data/voc11' ;
%opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn32s-voc11/net-epoch-5.mat' ;
opts.modelPath = '/net/mraid11/export/data/amirro/matconv_data/pascal-fcn8s-tvg-dag.mat';
%opts.modelFamily = 'matconvnet' ;
opts.modelFamily = 'ModelZoo' ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = true ;
opts.vocAdditionalSegmentationsMergeMode = 2 ;
% opts = vl_argparse(opts, varargin) ;
opts.useGpu = 1;
% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = vocSetup('dataDir', opts.dataDir, ...
    'edition', opts.vocEdition, ...
    'includeTest', false, ...
    'includeSegmentation', true, ...
    'includeDetection', false) ;
  if opts.vocAdditionalSegmentations
    imdb = vocSetupAdditionalSegmentations(...
      imdb, ...
      'dataDir', opts.dataDir, ...
      'mergeMode', opts.vocAdditionalSegmentationsMergeMode) ;
  end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get validation subset
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% Compare the validation set to the one used in the FCN paper
% valNames = sort(imdb.images.name(val)') ;
% valNames = textread('data/seg11valid.txt', '%s') ;
% valNames_ = textread('data/seg12valid-tvg.txt', '%s') ;
% assert(isequal(valNames, valNames_)) ;

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

switch opts.modelFamily
  case 'matconvnet'
    net = load(opts.modelPath) ;
   
        net = dagnn.DagNN.loadobj(net.net) ;
   
    net.mode = 'test' ;
    for name = {'objective', 'accuracy'}
      net.removeLayer(name) ;
    end
    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
    predVar = net.getVarIndex('prediction') ;
    inputVar = 'input' ;
    imageNeedsToBeMultiple = true ;

  case 'ModelZoo'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('upscore') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;

  case 'TVG'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('coarse') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;
end

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------


%% run on fra_db, too
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath utils;
%%
load ~/code/mircs/fra_db_2015_10_08.mat
net.move('gpu');
load(opts.imdbPath);
load('/home/amirro/storage/mircs_18_11_2014/s40_fra.mat');
%%
imgDir = '/home/amirro/storage/data/Stanford40//JPEGImages/';
numGpus = 0 ;
confusion = zeros(21) ;
outPath='~/storage/fcn_tvg_fra/';ensuredir(outPath)
for i = 1:1:length(s40_fra)
    imId =i            
    rgbPath = j2m(imgDir,s40_fra(i),'.jpg');
    outFile = j2m(outPath,s40_fra(i))
%     if exist(outFile,'file'),continue,end    
    % Load an image and gt segmentation
    rgb = vl_imreadjpeg({rgbPath}) ;
    rgb = rgb{1} ;
    %
    %     %   rgb = imcrop(rgb/255)*255;
    %     % Subtract the mean (color)
    %     im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage);
    %     net.meta.classes.name
    [pred_full,scores] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,'coarse',rgb,1,true,net.meta.classes.name);
    dpc
%     clf;
%     imagesc2(scores(:,:,15));drawnow;dpc
%     save(outFile,'scores');    
%     dpc
    %     [scores,pred] = applyNet(net,rgb,imageNeedsToBeMultiple,inputVar);    
end
%%

numGpus = 0 ;
confusion = zeros(21) ;
net.move('gpu');
for i = 1:numel(val)
  imId = val(i) ;
  rgbPath = sprintf(imdb.paths.image, imdb.images.name{imId}) ;
  labelsPath = sprintf(imdb.paths.classSegmentation, imdb.images.name{imId}) ;

  % Load an image and gt segmentation
  rgb = vl_imreadjpeg({rgbPath}) ;
  rgb_orig = rgb{1} ;
  rgb = imResample(rgb_orig,1,'bilinear');
  anno = imread(labelsPath) ;
  lb = single(anno) ;
  lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg

%   rgb = imcrop(rgb/255)*255;
  % Subtract the mean (color)
  im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;
  
  %im = imresize(im,4);

  % Soome networks requires the image to be a multiple of 32 pixels
  if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
  else
    im_ = im ;
  end

  
  
  net.eval({inputVar, gpuArray(im_)}) ;
  %scores_ = gather(net.vars(predVar).value) ;
  scores_ = gather(net.vars(end).value) ;
  scores_ = imResample(scores_,size2(rgb_orig),'bilinear');
  [~,pred_] = max(scores_,[],3) ;

  if imageNeedsToBeMultiple
    pred = imresize(pred_, sz, 'method', 'nearest') ;
  else
    pred = pred_ ;
  end
  
  % Accumulate errors
  ok = lb > 0 ;
  confusion = confusion + accumarray([lb(ok),pred(ok)],1,[21 21]) ;

  % Plots
  if mod(i - 1,10) == 0 || i == numel(val)
    [iu, miu, pacc, macc] = getAccuracies(confusion) ;
    fprintf('IU ') ;
    fprintf('%4.1f ', 100 * iu) ;
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
            100*miu, 100*pacc, 100*macc) ;

    figure(1) ; clf;
    imagesc(normalizeConfusion(confusion)) ;
    axis image ; set(gca,'ydir','normal') ;
    colormap(jet) ;
    drawnow ;

    % Print segmentation
    figure(100) ;clf ;
    displayImage(rgb/255, lb, pred) ;
    drawnow ;

    %   Save segmentation
    %   imname = strcat(opts.results,sprintf('/%s.png',imdb.images.name{subset(i)}));
    %   imwrite(pred,labelColors(),imname,'png');
  end
end

% -------------------------------------------------------------------------
% function nconfusion = normalizeConfusion(confusion)
% % -------------------------------------------------------------------------
% % normalize confusion by row (each row contains a gt label)
% nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2))) ;

% -------------------------------------------------------------------------
% function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% % -------------------------------------------------------------------------
% pos = sum(confusion,2) ;
% res = sum(confusion,1)' ;
% tp = diag(confusion) ;
% IU = tp ./ max(1, pos + res - tp) ;
% meanIU = mean(IU) ;
% pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
% meanAccuracy = mean(tp ./ max(1, pos)) ;
% 
% % -------------------------------------------------------------------------
% function displayImage(im, lb, pred)
% % -------------------------------------------------------------------------
% subplot(2,2,1) ;
% image(im) ;
% axis image ;
% title('source image') ;
% 
% subplot(2,2,2) ;
% image(uint8(lb-1)) ;
% axis image ;
% title('ground truth')
% 
% cmap = labelColors() ;
% subplot(2,2,3) ;
% image(uint8(pred-1)) ;
% axis image ;
% title('predicted') ;
% 
% colormap(cmap) ;

% -------------------------------------------------------------------------
% function cmap = labelColors()
% % -------------------------------------------------------------------------
% N=21;
% cmap = zeros(N,3);
% for i=1:N
%   id = i-1; r=0;g=0;b=0;
%   for j=0:7
%     r = bitor(r, bitshift(bitget(id,1),7 - j));
%     g = bitor(g, bitshift(bitget(id,2),7 - j));
%     b = bitor(b, bitshift(bitget(id,3),7 - j));
%     id = bitshift(id,-3);
%   end
%   cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
% end
% cmap = cmap / 255;
