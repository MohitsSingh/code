function regions = shiftRegions(regions,bbox,I)
wasCell = iscell(regions);
if (~wasCell), regions = {regions}; end
% in the rare even that the bounding box is not inside I,
bbox2 = clip_to_image(bbox,I);
topLeft = bbox2(1:2)-bbox(1:2)+1;
regionBox = [topLeft,topLeft+bbox2(3:4)-bbox2(1:2)];
% regions1 = cellfun2(@(x)  padarray(padarray(x, bbox([2 1])-1,0,'pre'),size2(I)-bbox([4 3]),0,'post'),regions);
for k = 1:length(regions)
    Z = zeros(size2(I));
    Z(bbox2(2):bbox2(4),bbox2(1):bbox2(3)) = regions{k}(regionBox(2):regionBox(4),regionBox(1):regionBox(3));
    regions{k} = Z;
end

if (~wasCell), regions = regions{1}; end