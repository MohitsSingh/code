function hist = getImageDescriptor(model, im,mask,curFeats)
% -------------------------------------------------------------------------

if (iscell(mask)) % multiple masks for one image,
    %concatenate features from all of them
    hist = {};
    % if it's going to happen a lot, call sub2ind only once.
    %     curFeats.frames1
    
    curFeats.frames1 = sub2ind(dsize(im,1:2),...
        curFeats.frames(2,:),curFeats.frames(1,:));
    curFeats.frames = double(curFeats.frames);
    % make an image of the quantized descriptors...
    scales = unique(curFeats.frames(4,:));
    dImage = zeros([dsize(im,1:2),length(scales)]);
    for iScale = 1:length(scales)
        curFrames = curFeats.frames(:,curFeats.frames(4,:)==scales(iScale));
        subs_ = [curFrames(2,:)',curFrames(1,:)',ones(size(curFrames,2),1)*iScale];
        dImage(sub2ind2(size(dImage),subs_)) =... 
            curFeats.binsa(curFeats.frames(4,:)==scales(iScale));        
    end
    curFeats.dImage = dImage;
    for k = 1:length(mask)
        hist{k} = getImageDescriptor(model, im,mask{k},curFeats);
    end
    hist = cat(2,hist{:});
    return;
else
    frames1 = [];
end

% im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

if (nargin < 3)
    mask = ones(height,width);
else % check if it's a full mask or a vector of indices
    if (size(mask,2)==1)
        mask_ = zeros(height,width);
        mask_(mask) = 1;
        mask = mask_;
    end
end

%if (numel(mask)==4)
if (size(mask,1)==1 && size(mask,2)==4) % bbox
    mask_ = mask;
    mask = zeros(height,width);
    mask(mask_(2):mask_(4),mask_(1):mask_(3)) = 1;
end

binsa = [];
if (nargin == 4 && ~isempty(curFeats))
    frames = curFeats.frames;
    dImage = curFeats.dImage;
    if (isfield(curFeats,'frames1'))
        frames1 = curFeats.frames1;
    end
    descrs = curFeats.descrs;
    binsa = curFeats.binsa;
else
    [frames, descrs] = vl_phow(im, model.phowOpts{:}) ;
end

% mask(1:end/4,:) = 0;
% get PHOW features

resplit = true;

if (isempty(binsa))
        D = l2(single(descrs'),model.vocab');
        [~,binsa] = min(D,[],2);
        binsa = row(binsa);
end


if (~isempty(frames1))
    weights = (mask(frames1));
else
    weights = (mask(sub2ind(size(mask),frames(2,:),frames(1,:))));
end
%weights = double(weights);

ymin = 1;
xmin = 1;
xmax = width;
ymax = height;

if (resplit)
    [ii,jj] = find(mask);
    xmin = min(jj);
    xmax = max(jj);
    ymin = min(ii);
    ymax = max(ii);
end
binsa = double(binsa);

isInside = frames(1,:) >  xmin & frames(2,:) > ymin & frames(1,:) < xmax &...
        frames(2,:) < ymax;
       
frames = double(frames(:,isInside));
dImage = dImage.*repmat(mask,[1 1 size(dImage,3)]);
for i = 1:length(model.numSpatialX)
    L1 = linspace(xmin,xmax,model.numSpatialX(i)+1);
    L2 =linspace(ymin,ymax,model.numSpatialY(i)+1);
%     [binsx,binsy] = getBins(dImage,L1,L2);
    
%     binsx = vl_binsearch(L1, frames(1,:))' ;
%     binsy = vl_binsearch(L2, frames(2,:))';
%     bins = sub2ind2([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
%         [binsy,binsx,binsa(isInside)']) ;
    hist = zeros(model.numSpatialY(i) , model.numSpatialX(i) , numWords) ;
    for iX = 1:length(L1)-1
        for iY = 1:length(L2)-1
            curBins = dImage(L1(iY):L1(iY+1)-1, L1(iX):L1(iX+1)-1,:);
            curBins = curBins(curBins>0);
            hist(iX,iY,:) = vl_binsum(hist(iX,iY,:),curBins);
        end
    end
    hist = hist(:);
    
%     hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
%     hist = vl_binsum(hist, double(weights(isInside)), bins) ;
    hists{i} = single(hist / sum(hist)) ;
end

hist = cat(1,hists{:}) ;

hist = hist / sum(hist) ;

if (any(isnan(hist(:))))
    a = 2342;
end
