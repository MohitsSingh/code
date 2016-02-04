% function my_fcnTest(varargin)
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths


opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-occlusion/net-epoch-50.mat';
opts.modelFamily = 'matconvnet' ;

opts.nClasses = 0;


load polys
% load ~/code/mircs/images_and_face_obj_full.mat;


opts.useGpu = 0;
% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------
 
% Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations

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

numGpus = 0 ;
nClasses = imdb.nClasses+1;
confusion = zeros(nClasses) ;
net.move('gpu');
%%
for i = 1:1:numel(test)
  imId = test(i) ;
    
  y = getBatch_action_obj(imdb, imId,'labelOffset',1); 
  rgb = y{2};
  anno = y{4};  
  lb = single(anno) ;
 
  im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;
 
  % Soome networks requires the image to be a multiple of 32 pixels
  if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
  else
    im_ = im ;
  end
    
  net.eval({inputVar, gpuArray(im_)}) ;

  scores_ = gather(net.vars(net.getVarIndex('prediction')).value);

  [~,pred_] = max(scores_,[],3) ;

  if imageNeedsToBeMultiple
    pred = imresize(pred_, sz, 'method', 'nearest') ;
  else
    pred = pred_ ;
  end

  % Accumulate errors
  ok = lb > 0 ;
  confusion = confusion + accumarray([lb(ok),pred(ok)],1,[nClasses nClasses]) ;

  % Plots
  if mod(i - 1,1) == 0 || i == numel(val)
      [iu, miu, pacc, macc] = getAccuracies(confusion) ;
      fprintf('IU ') ;
      fprintf('%4.1f ', 100 * iu) ;
      fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
          100*miu, 100*pacc, 100*macc) ;
      
      figure(1) ;
      clf;
      mm  =2;
      nn = 3;
      subplot(mm,nn,1);
      imagesc2(rgb/255); 
      axis image ; 
      %set(gca,'ydir','normal') ;
      colormap(jet) ;
      %     drawnow ;
      
      % Print segmentation
      %     figure(100) ;clf ;
      subplot(mm,nn,2);
      %displayImage(rgb/255, lb, pred) ;
      imagesc(uint8(pred-1)) ;
      axis image ;
      title('predicted') ;
      subplot(mm,nn,3);
        imagesc2(lb);

      subplot(mm,nn,4);
      imagesc2(sc(cat(3,scores_(:,:,2),rgb/255),'prob'));title('1');      
      subplot(mm,nn,5);
      imagesc2(sc(cat(3,scores_(:,:,3),rgb/255),'prob'));title('2');
      subplot(mm,nn,6);
      imagesc2(sc(cat(3,scores_(:,:,1),rgb/255),'prob')); title('bkg');
       drawnow ;
      
      pause
      continue
      %SZ = 513;
      SZ = size(rgb,1);
      rgb_p = rgb/255;
      rgb_p = imresize(rgb_p,[SZ,SZ],'bilinear');
      imwrite(rgb_p,'d1/1.ppm');
      data = zeros([SZ,SZ,3],'single');
      %data = -inf([SZ,SZ,3],'single');
      data(:,:,1:size(scores_,3)) = imResample(single(scores_),[SZ SZ]);
      SS = normalise(scores_(:,:,2));
      data(:,:,2) = imResample(SS,[SZ SZ]);
      save('feats/1_blob_0.mat','data','-v6');
     
      
      
%       A = imread('~/storage/data/Stanford40/JPEGImages/applauding_011.jpg');
%       A = imResample(A,[513 513]);
%       imwrite(A,'d1/1.ppm');      
%       data = zeros(513,513,21,'single');
%       data(50:100,50:100,16) = 1;
%       save('feats/1.mat','data','-v6');
    %   Save segmentation
    %   imname = strcat(opts.results,sprintf('/%s.png',imdb.images.name{subset(i)}));
    %   imwrite(pred,labelColors(),imname,'png');
  end
end



