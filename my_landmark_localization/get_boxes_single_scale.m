function boxes = get_boxes_single_scale(wSize, cellSize,I,dense)
wSize_cell = wSize/cellSize;
imgSize_cell = size2(I)/cellSize;
% assume zero padding, for now.
%     tic
%     for z = 1:10

[xx,yy] = meshgrid(1:imgSize_cell(2)-wSize_cell+1,1:imgSize_cell(2)-wSize_cell+1);
nFeats = numel(xx);
boxes = zeros(nFeats,4);
xx = xx(:);
yy = yy(:);

for t = 1:length(xx)
    x = xx(t); y = yy(t);
    curRect = [(x-1)*cellSize+1 (y-1)*cellSize+1 (x+wSize_cell-1)*cellSize (y+wSize_cell-1)*cellSize];
    boxes(t,:) = curRect;
    %     X(:,:,:,t) = U(y:y+wSize_cell-1,x:x+wSize_cell-1,:);
end
% X = reshape(X,[],nFeats);
end