function [res,rects] = extractDNNFeats_tiled(imgs,net,tiles,layers,prepareSimple)
% same as extractDNNFeats, but splits each image into [r x c] tiles
% before extracting features.

if (nargin < 5)
    prepareSimple = false;
end
if nargin < 4
    layers = [16 19];
end
if (~iscell(imgs))
    imgs = {imgs};
end
% split into batches of 256...
res = struct('layer_num',{},'x',{});
% split each image into tiles and feed back into cell array.
imgs_tiled = {};
rects = {};
for iImg = 1:length(imgs)    
%     if (iscell(tiles))
%         curTiles = tiles(iImg);
%     else
%         curTiles = tiles;
%     end
    [curTiles,curRects] = splitToTiles(imgs{iImg},tiles);
    imgs_tiled{iImg} = row(curTiles);
    rects{iImg} = curRects;
end
imgs = cat(2,imgs_tiled{:});
res = extractDNNFeats(imgs,net,layers,prepareSimple);
