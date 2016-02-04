function y = getBatch_action_obj_coco(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [512, 512] - 128 ;
% opts.imageSize = [384, 384] - 128 ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.rgbMean = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.labelStride = 1 ;
opts.labelOffset = 0 ;
opts.classWeights = ones(1,imdb.nClasses+1,'single');
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.useGpu = false ;
opts.useFlipping = true;
opts.doScaling = true;
opts = vl_argparse(opts, varargin);

if opts.prefetch
    % to be implemented
    ims = [] ;
    labels = [] ;
    return ;
end

if ~isempty(opts.rgbVariance) && isempty(opts.rgbMean)
    opts.rgbMean = single([128;128;128]) ;
end
if ~isempty(opts.rgbMean)
    opts.rgbMean = reshape(opts.rgbMean, 1,1,[]) ;
end

% space for images
ims = zeros(opts.imageSize(1), opts.imageSize(2), size(opts.rgbMean,3), ...
    numel(images)*opts.numAugments, 'single') ;

% space for labels
lx = opts.labelOffset : opts.labelStride : opts.imageSize(2) ;
ly = opts.labelOffset : opts.labelStride : opts.imageSize(1) ;
labels = zeros(numel(ly), numel(lx), 1, numel(images)*opts.numAugments, 'single') ;
classWeights = [0 opts.classWeights(:)'] ;

im = cell(1,numel(images)) ;

si = 1 ;
dataDir = imdb.dataDir;
% find out if this is train or validation
dataType = 'train2014';
curData = imdb.coco_train;
if imdb.set(images(1)) ~= 1 % val
    curData = imdb.coco_val;
    dataType = 'val2014';
end
for i=1:numel(images)
    
    k = images(i);
    imgId = imdb.images_ids(k);
    img = curData.loadImgs(imgId);
    rgb = single(imread(sprintf('%s/images/%s/%s',dataDir,dataType,img.file_name)));
    rgb = single(rgb);
    
    % get the annotation
    annIds = curData.getAnnIds('imgIds',imgId,'iscrowd',[]);
    anns = curData.loadAnns(annIds);
    
    
    %% render the annotations to masks.
    n=length(anns);
    S={anns.segmentation};  k=0;
    anno = zeros(size2(rgb));
    for ii = 1:n
        if(isstruct(S{ii}))
            M=double(MaskApi.decode(S{ii}));
        else for j=1:length(S{ii})
                P=S{ii}{j}+.5; k=k+1;
                M = poly2mask2([P(1:2:end);P(2:2:end)]',size(anno));
                %hs(k)=fill(P(1:2:end),P(2:2:end),C,pFill{:});
            end
        end
        anno(M>0) = anns(ii).category_id;
    end
    
    %%
    
    %
    %     for t = 1:length(anns)
    %         clf; imagesc2(rgb/255);
    %         curData.showAnns(anns(t));
    %         title(categoryMap(anns(t).category_id).name)
    %         dpc
    %     end
    %
    %     for i = 1:length(anns)
    %
    %     end
    %
    % acquire image
    %   if isempty(im{i})
    %     rgbPath = sprintf(imdb.paths.image, imdb.images.name{images(i)}) ;
    %     labelsPath = sprintf(imdb.paths.classSegmentation, imdb.images.name{images(i)}) ;
    %     rgb = vl_imreadjpeg({rgbPath}) ;
    %     rgb = rgb{1} ;
    %     anno = imread(labelsPath) ;
    %   else
    %     rgb = im{i} ;
    %   end
    %   if size(rgb,3) == 1
    %     rgb = cat(3, rgb, rgb, rgb) ;
    %   end
    
    % crop & flip
    h = size(rgb,1) ;
    w = size(rgb,2) ;
    for ai = 1:opts.numAugments
        sz = opts.imageSize(1:2) ;
        scale = max(h/sz(1), w/sz(2)) ;
        if opts.doScaling
            scale = scale .* (1 + (rand(1)-.5)/5) ;
        end
        
        sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2) ;
        sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2) ;
        if opts.useFlipping
            if rand > 0.5, sx = fliplr(sx) ; end
        end
        
        okx = find(1 <= sx & sx <= w) ;
        oky = find(1 <= sy & sy <= h) ;
        if ~isempty(opts.rgbMean)
            ims(oky,okx,:,si) = bsxfun(@minus, rgb(sy(oky),sx(okx),:), opts.rgbMean) ;
        else
            ims(oky,okx,:,si) = rgb(sy(oky),sx(okx),:) ;
        end
        
        
        if (any(anno(:)))
            tlabels = zeros(sz(1), sz(2), 'uint8') + 255 ;
            tlabels(oky,okx) = anno(sy(oky),sx(okx)) ;
            tlabels = single(tlabels(ly,lx)) ;
            tlabels = mod(tlabels + 1, 256) ; % 0 = ignore, 1 = bkg
            labels(:,:,1,si) = tlabels ;
        end
        si = si + 1 ;
    end
end
if opts.useGpu
    ims = gpuArray(ims) ;
end
y = {'input', ims, 'label', labels} ;
