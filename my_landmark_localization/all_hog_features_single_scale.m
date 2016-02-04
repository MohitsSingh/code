function [X,boxes] = all_hog_features_single_scale(wSize, cellSize,I)
wSize_cell = wSize/cellSize;
imgSize_cell = size2(I)/cellSize;
% assume zero padding, for now.
%     tic
%     for z = 1:10
U = fhog2(I,cellSize);
if isscalar(wSize_cell)
    wSize_cell = [wSize_cell wSize_cell];
end
h = wSize_cell(1);
w = wSize_cell(2);
[xx,yy] = meshgrid(1:imgSize_cell(2)-h+1,1:imgSize_cell(1)-w+1);
nFeats = numel(xx);
X = zeros(h,w,31,nFeats);
boxes = zeros(nFeats,4);
xx = xx(:);
yy = yy(:);

for t = 1:length(xx)
    x = xx(t); y = yy(t);
    curRect = [(x-1)*cellSize+1 (y-1)*cellSize+1 (x+w-1)*cellSize (y+h-1)*cellSize];
    boxes(t,:) = curRect;
    X(:,:,:,t) = U(y:y+h-1,x:x+w-1,:);
end
X = reshape(X,[],nFeats);
end