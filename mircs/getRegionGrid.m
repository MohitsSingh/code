function [regions,sz,jump] = getRegionGrid(I,sz,jump)
    width = size(I,2);
    height = size(I,1);
    if (isempty(sz))
        sz = floor(mean(dsize(I,1:2))/6);
        sz = [sz sz];
    end
    if (nargin < 3)
%         jump = mean(dsize(I,1:2))/30;
        jump = floor(mean(sz)/6);
    end
    x = 1:jump:width-sz(2);
    y = 1:jump:height-sz(1);
    [x,y] = meshgrid(x,y);
    %figure,imshow(I); hold on; plot(x,y,'g.')
    x = x(:);
    y = y(:);
    regions = [x y x+sz(2) y+sz(1)];
    regions = mat2cell2(regions,[size(regions,1) 1]);
%     
%     regions_ = vl_colsubset(regions',200)';
%     
%     figure,imshow(I); hold on; plotBoxes2(regions_(:,[2 1 4 3]),'g');
%     
end