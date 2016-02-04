function polys = rotate_bbs(rects,I,theta,aroundImageCenter)
if (isempty(rects))
    polys = [];
end
sz_ = size(I);
if (nargin < 4)
    aroundImageCenter = true;
end
if (aroundImageCenter)
    centerPoint = fliplr((sz_(1:2)/2));
end
if (isscalar(theta))
    thetas = ones(size(rects,1),1)*theta;
else
    thetas = theta;
end
for k = 1:size(rects,1)
    curPts = box2Pts(rects(k,1:4));
    % shift to center, rotate, and shift back.
    %imshow(I); hold on; hold on;plot(curPts(:,1),curPts(:,2),'g-+')
    if (~aroundImageCenter)
        centerPoint = mean(curPts,1);
    end
    
    curPts = rotate_pts(curPts,-pi*thetas(k)/180,centerPoint);
    
    %     curPts_c = bsxfun(@minus,curPts,centerPoint);
    %     R = rotationMatrix(-pi*thetas(k)/180);
    %     curPts = bsxfun(@plus,curPts_c*R,centerPoint);
    %hold on; hold on;plot(curPts(:,1),curPts(:,2),'r-+')
    polys{k} = curPts;
end
end