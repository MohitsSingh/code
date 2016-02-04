function [dets] = detect_rotated2(I,classifier,cell_size,features,detection,...
    threshold,thetas,debug_)
dets = struct('theta',{},'polys',{},'scores',{},'rects',{},'class',{});
if (~exist('debug_','var'))
    debug = false;
end
sz = dsize(I,1:2);
newSize = ceil(max(sz)*1.1);
mask = ones(dsize(I,1:2));
padSize = ceil((newSize-sz)/2);
I = max(0,min(1,I));
I_padded = padarray(I,padSize,0,'both');
mask_padded = padarray(mask,padSize,0,'both');
weights = {classifier.weights};
bias = {classifier.bias};
obj_size = {classifier.object_sz};
for k = 1:length(thetas)
    curTheta = thetas(k);
    fprintf('%d...',curTheta);
    R = rotationMatrix([0 0 1],curTheta*pi/180);
    I_rot = imtransform2(I_padded,R,'method','cubic','bbox','crop');
    I_rot = max(0,min(1,I_rot));
    mask_rot = imrotate(mask_padded,curTheta,'nearest','crop');
    all_rects = detect2(I_rot,weights,bias,...
        obj_size,cell_size,features,detection,threshold,mask_rot,debug_);
    curRes = struct('theta',{},'polys_rotated',{},'scores',{},'rects',{},'class',{});
    for iClass = 1:length(weights)
        rects = all_rects{iClass};
        if (isempty(rects))
            continue;
        end
        rects(:,3:4) = rects(:,3:4) + rects(:,1:2);
        polys = rotate_bbs(rects,I_padded,curTheta);
        polys = cellfun(@(x) bsxfun (@minus, x, padSize([2 1])), polys,'UniformOutput',false);
        polys = polys(:); % column
        rects = bsxfun(@minus, rects, [padSize([2 1 2 1]) 0]);
        curRes(iClass).theta = ones(size(rects,1),1)*curTheta;
        curRes(iClass).polys_rotated = polys;
        curRes(iClass).scores = rects(:,end);
        curRes(iClass).rects = rects;
        curRes(iClass).class = ones(size(rects,1),1)*iClass;
    end
    dets_{k} = curRes(:);
end
dets_ = cat(1,dets_{:});
if (isempty(dets_))
    return;
end
    
theta = cat(1,dets_.theta);
polys = cat(1,dets_.polys_rotated);
scores = cat(1,dets_.scores);
rects = cat(1,dets_.rects);
class = cat(1,dets_.class);

theta = mat2cell2(theta,size(theta,1));
% polys = mat2cell2(polys,size(polys,1));
scores = mat2cell2(scores,size(scores,1));
rects = mat2cell2(rects,[size(rects,1),1]);
class = mat2cell2(class,size(class,1));

dets = struct('theta',theta,'polys',polys,'scores',scores,'rects',rects,'class',class);
fprintf('\n');
end
