function [ hists , hogs, rects,q] = bowScan( conf,model, img,nn)
%BOWSCAN Summary of this function goes here
%   Detailed explanation goes here
% divide the image into sectors and return the histogram of each
% sub-sector.
nSectors = [3 3];
n = prod(nSectors);
q = reshape(1:n,nSectors);
q = imresize(q,[size(img,1),size(img,2)],'nearest');
img_ = {};
sz = [40 40];
conf.features.vlfeat.cellsize = 8;
rects = {};
for k = 1:n
    if (nargin == 4 && k ~= nn)
        continue;
    end
    [row,col] = find(imdilate(q==k,ones(5)));
%             imagesc(bsxfun(@times,imdilate(q==k,ones(5)),im2double(img)));
%             pause;
    [ymin,ymax,xmin,xmax] = deal(min(row),max(row),min(col),max(col));
    img_{k} = imresize(img(ymin:ymax,xmin:xmax,:),sz);    
    rects{k} = [xmin xmax ymin ymax];
end

hists = getBOWFeatures(conf,model,img_,[]);
hogs = imageSetFeatures2(conf,img_,true,sz);
rects = cat(1,rects{:});
end

