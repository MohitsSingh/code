function [orthLines] = isDrinking(im,faceLines,probMap)
%ISDRINKING Summary of this function goes here
%   Detailed explanation goes here


debug_ = true;
if (debug_)
    close all;
end
im = imresize(im,2);faceLines = faceLines * 2;
E = edge(im2double(rgb2gray2(im)),'canny');
minLength = 5;
[edgelist] = edgelink(E, minLength);
if (debug_)
    figure,imagesC(im);
    hold on;
end
tol = 3;
seglist = lineseg(edgelist, tol);

edgeListImage= edgelist2image(edgelist, size(E));
% figure,imagesc(edgeListImage)
transformParams = [];
transformParams.xRange = -6:.5:6;
transformParams.yRange = -4:.5:4;
transformParams.scaleRange = .8:.1:1.2;
%  transformParams.scaleRange = 1;;
% transformParams.scaleRange = 1;
probMap = imresize(probMap,2);

p_edge = edge(probMap,'canny');
p_edge = p_edge.*double(probMap >=.2);
% [y_min,x_min]  = fitCurveToImage(edge(probMap,'canny'),faceLines);
[y_min,x_min]  = fitCurveToImage_old(E,faceLines,transformParams);
xy_  =[x_min y_min];
faceLines = [xy_(1:end-1,:) xy_(2:end,:)];
% h =  drawedgelist(seglist, size(E), 2, 'rand', 1);

segs = seglist2segs(seglist);
segs = segs(:,[2 1 4 3]);

out = lineSegmentIntersect(faceLines,segs);


if (debug_)
    imagesc(im); hold on;
    line([faceLines(:,1)';faceLines(:,3)'],[faceLines(:,2)';faceLines(:,4)'],'Color','r');
    line([segs(:,1)';segs(:,3)'],[segs(:,2)';segs(:,4)'],'Color','g');
end

f = find(out.intAdjacencyMatrix);
% figure,imagesc(out.intAdjacencyMatrix)

v1 = segs2vecs(faceLines);
v2 = segs2vecs(segs);
v1 = normalize_vec(v1')';
v2 = normalize_vec(v2')';

theta = acosd(v1*v2');

isOrthogonal = abs(theta(f)-90) <= 30;
if (debug_)
    hold on;
    scatter(out.intMatrixX(f(isOrthogonal)),out.intMatrixY(f(isOrthogonal)),[],'gs');
end
[ii,jj] = find(out.intAdjacencyMatrix);
orthLines = segs(unique(jj),:);
end