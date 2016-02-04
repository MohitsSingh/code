function [res] = splitToTiles(I,tiles)
%UNTITLED2 split image into tiles(i,1) vertical tiles and tiles(i,2) horizontal
% tiles, for each row i for tiles.
% the total number of resultant tiles is :sum(prod(tiles,2)) total blocks
%   Detailed explanation goes here
res = [];

% tiles = [1 1;2 2;3 3;4 4];

for iTiles = 1:size(tiles,1)
    curTiles = tiles(iTiles,:);
    tilesY = curTiles(2);
    tilesX = curTiles(1);
    
    tileSizeY = size(I,1)/tilesY;
    tileSizeX = size(I,2)/tilesX;
    
    rects = zeros(prod(curTiles),4);
    t = 0;
    for iX = 1:tilesX
        for iY = 1:tilesY
            t = t+1;
            curRect = [(iX-1)*tileSizeX+1,(iY-1)*tileSizeY+1,iX*tileSizeX,iY*tileSizeY];
            rects(t,:) = curRect;
        end
    end
    rects = round(clip_to_image(rects,I));
    res = [res,row(multiCrop2(I,rects))];
end
end

