function y = getBatch_p_theta_and_scale(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [224 224];
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
opts.useFlipping = false;
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

im = cell(1,numel(images)) ;

si = 1 ;

for i=1:numel(images)
    
    k = images(i);
    rgb = single(imdb.images_data{k});
    %     rgb = imResample(rgb,opts.imageSize,'bilinear');
    %     rgb = imResample(rgb,2,'bilinear');
    
    anno = imdb.labels(k,:);
    
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
        
        
        
        labels(:,:,1,si) = anno ;
        
        si = si + 1 ;
    end
end
if opts.useGpu
    ims = gpuArray(ims) ;
end
y = {'input', ims, 'label_theta', labels(1,1,1,:),'label_scale',labels(1,2,1,:)} ;
