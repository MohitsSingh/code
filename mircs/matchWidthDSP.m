function [warp21,warp21_smooth,vx,vy] = matchWidthDSP(im1,im2,pca_basis,sift_size)
if (nargin < 4)
    sift_size = 4;
end
[sift1, bbox1] = ExtractSIFT_WithPadding(im1, pca_basis, sift_size);
[sift2, bbox2] = ExtractSIFT_WithPadding(im2, pca_basis, sift_size);
% % % % im1 = im1(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
% % % % im2 = im2(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);
% anno1 = anno1(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
% anno2 = anno2(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);

% Match
tic;
[vx,vy] = DSPMatch(sift1, sift2);
[x,y] = meshgrid(1:size(vx,2),1:size(vy,1));
T = cp2tform([x(:)+vx(:) y(:)+vy(:)],[x(:) y(:)],'similarity');
warp21_smooth = imtransform(im2,T,'bilinear','XData',[1 size(im2,2)],'YData',[1 size(im2,1)]);
t_match = toc;

% Evaluation
% [seg, acc] = TransferLabelAndEvaluateAccuracy(anno1, anno2, vx, vy);
% acc.time = t_match;

% Warping
warp21=warpImage(im2double(im2),vx,vy); % im2 --> im1
