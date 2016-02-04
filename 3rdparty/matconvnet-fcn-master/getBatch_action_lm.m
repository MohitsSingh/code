function y = getBatch_action_lm(imdb, images, varargin)
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
  opts.rgbMean = reshape(opts.rgbMean, [1 1 3]) ;
end

% space for images
ims = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
  numel(images)*opts.numAugments, 'single') ;

% space for labels
lx = opts.labelOffset : opts.labelStride : opts.imageSize(2) ;
ly = opts.labelOffset : opts.labelStride : opts.imageSize(1) ;
labels = zeros(numel(ly), numel(lx), 1, numel(images)*opts.numAugments, 'single') ;
classWeights = [0 opts.classWeights(:)'] ;

im = cell(1,numel(images)) ;

si = 1 ;
LUT = [1 1 2 2 2 3 4]; 
lutToShow={};
for u = 1:length(LUT)
    lutToShow{u} = num2str(LUT(u));
end
% lutToShow = {'1' '1' '2' '2' '2' '3' '4'}; 
s = strel('disk',11);
for i=1:numel(images)
    
    k = images(i);
    rgb = single(imdb.images_data{k});
    sz_orig = size2(rgb);
    rgb = imResample(rgb,opts.imageSize,'bilinear');
    anno = [];
    
    if (isfield(imdb,'labels'))
        %anno = imdb.labels{k};
        curLabels = imdb.labels{k};
        L = curLabels{2};
        L = bsxfun(@times,L(:,1:2),size2(rgb)./sz_orig);
        %   for t = 1:length(images)
        %       t
%              clf; imagesc2(rgb/255);
%              plotPolygons(L,'g+','LineWidth',2)
%                  showCoords(L,lutToShow);            
        anno = zeros(size2(rgb));
        %       L = landmarks{t};
        %      1 2 3 4 5 6 7
        
        for u = 1:size(L,1)
            if any(L(u,:)<0),continue,end
%             if u<=2
%                u_=1;
%             else
%                u_ = u-1;
%             end
            anno(round(L(u,2)),round(L(u,1))) = LUT(u);
        end        
        anno = imdilate(single(anno),s);
        
        M = curLabels{1};
        M=imResample(M,size2(rgb),'nearest');
        M(M>0) = M(M>0)+4;        
%         x2(anno)
%         x2(M);
        M(M==5 & anno > 0) = anno(M==5 & anno > 0);
        anno = M;
%       
%         anno_to_show = anno+1;
%         anno_to_show(end) = 13;
%         clf; imagesc(anno_to_show); 
%         colormap(distinguishable_colors(12));
%         lcolorbar({'none','eye','mouth','chin center','nose tip','face','hand','drink','smoke','blow','brush','phone'});
%         dummy=0;
        %       curLabel = single(imdilate(curLabel,ones(3)));
        %       masks{t} = curLabel;
        %       %dpc
        %   end
        %         anno = imResample(anno,opts.imageSize,'nearest');
    end
      
  % crop & flip
  h = size(rgb,1) ;
  w = size(rgb,2) ;
  for ai = 1:opts.numAugments
    sz = opts.imageSize(1:2) ;
    scale = max(h/sz(1), w/sz(2)) ;
    scale = scale .* (1 + (rand(1)-.5)/5) ;

    sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2) ;
    sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2) ;
    if rand > 0.5, sx = fliplr(sx) ; end

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
