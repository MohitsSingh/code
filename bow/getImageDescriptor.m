function hists = getImageDescriptor(model,mask,feats)
% -------------------------------------------------------------------------
% numWords = size(model.vocab, 2);
numWords = arrayfun(@(x) size(x.vocab, 2),model);
numWords = numWords(1); % TODO - currently assumed number of visual words same for all dictionaries...

width = size(feats{1},2) ;
height = size(feats{1},1) ;
% TODO - it is currently assumed numSpatialX is same for all models....
numSpatialX = model(1).numSpatialX;
numSpatialY = model(1).numSpatialY;
if (nargin < 3)
    mask = ones(height,width);
end

if (~iscell(mask))
    
    if (size(mask,2)==1) % check if it's a full mask or a vector of indices
        mask_ = zeros(height,width);
        mask_(mask) = 1;
        mask = mask_;
    elseif (size(mask,1)==1 && size(mask,2)==4) % bbox
        mask_ = mask;
        mask = zeros(height,width);
        mask(mask_(2):mask_(4),mask_(1):mask_(3)) = 1;
    end
    mask = {mask};
end

% special case - mask is -1.
if (length(mask)==1)
    if (isscalar(mask{1}))
        mask{1} = true(height,width);
    end
end

hists = {};
bowImages = feats;
for k = 1:length(bowImages)
    bowImages{k} = double(bowImages{k});
end
for k = 1:length(mask)
    curMask = mask{k};
    isbbox = false;
    
    if (size(curMask,2)==4) % bounding box
        [xmin,ymin,xmax,ymax] = deal(curMask(1),curMask(2),curMask(3),curMask(4));
        isbbox = true;
    else
        if (size(curMask,2)==1) % a list of indices
            mask_ = zeros(height,width);
            
            mask_(curMask) = 1;
            curMask = mask_;
        end
        [ii,jj] = find(curMask);
        xmin = min(jj);
        xmax = max(jj);
        ymin = min(ii);
        ymax = max(ii);
        m = repmat(curMask,[1 1 size(bowImages{1},3)]);
    end
    
%     xmin = double(xmin);
%     xmax = double(ymax);
%     ymin = double(xmin);
%     ymax = double(ymax);
    % %     if (max(ymax - ymin,xmax-xmin) < 15)
    % %         hists{k} = nan(sum(model.numSpatialY.*model.numSpatialX)*numWords,1);
    % % %         k
    % %         continue;
    % %     end
    % %
    
    for i = 1:length(numSpatialX)
        L1 = round(linspace(xmin,xmax,numSpatialX(i)+1));
        L2 = round(linspace(ymin,ymax,numSpatialY(i)+1));
        hist = zeros(numSpatialY(i) , numSpatialX(i) , numWords, length(bowImages));
        for iX = 1:length(L1)-1
            for iY = 1:length(L2)-1
                for iImg = 1:length(bowImages)
                    curBins = bowImages{iImg}(L2(iY):L2(iY+1)-1, L1(iX):L1(iX+1)-1,:);
                    if (~isbbox)
                        curBins = curBins.*m(L2(iY):L2(iY+1)-1, L1(iX):L1(iX+1)-1,:);
                    end
                    hist(iX,iY,:,iImg) = vl_binsum(hist(iX,iY,:,iImg),ones(size(curBins)),curBins);
                end
            end
        end
        %         hist = hist(:,:,1:end-1);
        hist = hist(:);
        hists_{i} = single(hist / sum(hist)) ;        
%         hists_{i} = single(hist); %TODO - above line was replaced by this one.
    end
    
    hist = cat(1,hists_{:}) ;
    
    hists{k} = hist / sum(hist) ;
end

hists = cat(2,hists{:});
