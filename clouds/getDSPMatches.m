function [xy_src,xy_dst] = getDSPMatches(I1,I2,resampleFactor,augmentY)
if nargin < 4
    augmentY = false;
end
I1 = imResample(I1,resampleFactor,'bilinear');
I2 = imResample(I2,resampleFactor,'bilinear');

% load ct101_pca_basis.mat pca_basis 


pca_basis = [];
sift_size = 4;

% [level, em] = graythresh(I1);

m1 = I1>10/255;
m2 = I2>10/255;
m1_o = m1;m2_o = m2;
bounds1 = region2Box(m1);
bounds1(1:2) = bounds1(1:2)-15;
bounds1(3:4) = bounds1(3:4)+15;
bounds2 = region2Box(m2);
bounds2(1:2) = bounds2(1:2)-15;
bounds2(3:4) = bounds2(3:4)+15;

I1 = cropper(I1,bounds1);
I2 = cropper(I2,bounds1);
m1 = cropper(m1,bounds1);
m2 = cropper(m2,bounds1);

% extract SIFT
[sift1, bbox1] = ExtractSIFT(I1, [], sift_size);
[sift2, bbox2] = ExtractSIFT(I2, [], sift_size);
I1 = I1(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
I2 = I2(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);
m1 = m1(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
m2 = m2(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);
% anno1 = anno1(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
% anno2 = anno2(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);
if augmentY 
    [xx,yy] = meshgrid(1:size(m1,2),1:size(m1,1));
    sift1 = cat(3,sift1,yy*1000);
    sift2 = cat(3,sift2,yy*1000);
end
[vx,vy] = DSPMatch(sift1, sift2);

vx = vx.*m1;
vy = vy.*m1;
% 

% add to the sift features another column with their y coordinate, to
% penalize different y's


vx = medfilt2(vx);
vy = medfilt2(vy);

[yy,xx] = find(m1);
inds = find(m1);

vx = vx(inds);
vy = vy(inds);
xy_src = [xx yy];
xy_dst = xy_src + [vx vy];



xy_src = xy_src/resampleFactor;
xy_dst = xy_dst/resampleFactor;

% [X,Y] = meshgrid(1:size(I1,2),1:size(I2,1));
% xy_src = [X(:) Y(:)];


% x2(I1); plotPolygons(xy_src,'r.')
% x2(I2); plotPolygons(xy_dst,'r.')

% match_plot_x(I1,I2,xy_src,xy_dst,50);

% x2(I1); plotPolygons(xy_src,'r.')
% x2(I2); plotPolygons(xy_dst,'r.')
xy_src = bsxfun(@plus,xy_src,bounds1(1:2)+bbox1([1 3])')-2;
xy_dst = bsxfun(@plus,xy_dst,bounds1(1:2)+bbox2([1 3])')-2;

goods = inMask(xy_src,m1_o) & inMask(xy_dst,m2_o);
xy_src = xy_src(goods,:);
xy_dst = xy_dst(goods,:);
