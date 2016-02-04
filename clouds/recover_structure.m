% recover the reconstuction

% rmpath('/home/amirro/code/3rdparty/piotr_toolbox/classify/');

function res = recover_structure(times_and_views,world_to_cam_samples,cameras,...
    matchingType)

if nargin < 4
    matchingType='sparse';
end
% % addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox');
% % vl_setup;
% % addpath(genpath('~/code/utils'))
matchedDir = '/home/amirro/code/clouds/matched_pairs';
origDir = '/home/amirro/code/clouds/Images_divided_by_maxValue';
transformationsDir = 'xforms';ensuredir(transformationsDir);
imgPattern = 'Image_T_%02.0f_A_%02.0f.png';
warpPattern = fullfile(matchedDir,'T_%02.0f_%02.0f_%02.0f.png');
flowPattern = fullfile(matchedDir,'T_%02.0f_%02.0f_%02.0f.png.mat');
res = struct('T',{},'view1',{},'view2',{},'xyz',{},'uvw',{},'mask',{});
N = 0;
for it = 1:size(times_and_views,1)-1
    it
    time1 = times_and_views(it,1);
    time2 = times_and_views(it+1,1);
    view1 = times_and_views(it,2);
    view2 = times_and_views(it+1,2);
    % for iTime = 1:length(times)
%     iTime
    %     for iView = 1:length(views)
    
%     curTime = times(iTime);
%     view1 = views(iView);
    view1Name = sprintf(imgPattern,time1,view1);
    view2Name = sprintf(imgPattern,time2,view2);
    view1Path = fullfile(origDir,view1Name);
    view2Path = fullfile(origDir,view2Name);    
    I1 = im2single(imread(view1Path));
    I2 = im2single(imread(view2Path));    
    
%     clf; imagesc2([I1 I2]);dpc;continue
    pts1 = world_to_cam_samples(view1).cam;
    pts2 = world_to_cam_samples(view2).cam;
    [fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
        pts1, pts2, 'Method', 'RANSAC', ...
        'NumTrials', 10000, 'DistanceThreshold', 0.1, 'Confidence', 99.99);
    
    inlierPoints1 = pts1(epipolarInliers, :);
    inlierPoints2 = pts2(epipolarInliers, :);
    
   % fix with a known offset.
   
%    pts1 = bsxfun(@plus,pts1,[-16 15]);
%    pts2 = bsxfun(@plus,pts2,[-16 15]);
    
    [t1, t2] = estimateUncalibratedRectification(fMatrix, ...
        inlierPoints1, inlierPoints2, size(I2));
                
    tform1 = projective2d(t1);
    tform2 = projective2d(t2);    
    I1Rect = imwarp(I1, tform1, 'OutputView', imref2d(size(I1)));
    
    t1_ = rectify_shearing(t1',t2',size2(I2));
    tform1_ = projective2d(t1*t1_'); % t1_*t1 , t1_'*t1 , t1*t1_
    I1Rect_ = imwarp(I2, tform1_, 'OutputView', imref2d(size(I2)));
%     x2(I1Rect); title('before'); x2(I1Rect_);title('after');
    I2Rect = imwarp(I2, tform2, 'OutputView', imref2d(size(I2)));    
    mask = (I1Rect>10/255);
    [X,Y] = meshgrid(1:size(I1Rect,1),1:size(I1Rect,2));
    if strcmp(matchingType,'sanity')
        % transform ground truth to warped images and back.
%        match_plot_x(I1,I2,inlierPoints1,inlierPoints2);
        xy_src = tform1.transformPointsForward(inlierPoints1);
        xy_dst = tform2.transformPointsForward(inlierPoints2);
        %match_plot_x(I1Rect,I2Rect,xy_src,xy_dst);
        %     x2(I1Rect);plotPolygons(xy_src,'r.')
        %     x2(I1);plotPolygons(xy_src,'r.')
%         xy_src = tform1.transformPointsInverse(xy_src);
%         xy_dst = tform2.transformPointsInverse(xy_dst);
    elseif strcmp(matchingType,'deep')
        flowPath = sprintf(flowPattern,time1,view1,view2);        
        L = load(flowPath);
        X_dst = X+L.u;
        Y_dst = Y+L.v;        
        mask = mask & abs(L.v) < 1; 
        debug_mask = false(size(mask));
        debug_factor = 2;
        debug_mask(1:debug_factor:end,1:debug_factor:end,:) = true;
        mask = mask & debug_mask;
        mask1 = false(size(mask));
        mask1(1:1:end,1:1:end) = true;
        mask = mask & mask1;
        xy_src = [X(mask) Y(mask)];
        xy_dst = [X_dst(mask) Y_dst(mask)];
        
        dst_mask = imdilate(I2Rect>10/255,ones(3));
        inds2 = sub2ind2(size(dst_mask),fliplr(round(xy_dst)));
        dst_mask = dst_mask(inds2);
        xy_src = xy_src(dst_mask,:);
        xy_dst = xy_dst(dst_mask,:);
        d = 50;        
%         x2(I1Rect*0);  plotPolygons(xy_src(1:d:end,:),'r.');
%         x2(I2Rect*0);  plotPolygons(xy_dst(1:d:end,:),'r.');        
        [fMatrix1, epipolarInliers1, status1] = estimateFundamentalMatrix(...
            xy_src, xy_dst, 'Method', 'RANSAC', ...
            'NumTrials', d, 'DistanceThreshold', 0.1, 'Confidence', 99.99);

        xy_src = xy_src(epipolarInliers1, :);
        xy_dst = xy_dst(epipolarInliers1, :);
        % remove points which do not fit an estimate of F

%         match_plot_h(I1Rect, I2Rect, inlierPoints1, inlierPoints2,50);
        
%         match_plot_x(I1Rect,I2Rect,xy_src,xy_dst);
        
        
%         plot = dst_mask(inds2);
        
    else
        [xy_src,xy_dst] = getSparseMatches(I1Rect,I2Rect);
    end
%     x2(I1Rect);plotPolygons(xy_src,'r.')
    xy_src = tform1.transformPointsInverse(xy_src);
%     x2(I1);plotPolygons(xy_src,'r.')
    xy_dst = tform2.transformPointsInverse(xy_dst);
    
%     match_plot(I1, I2, pts1, pts2,10);    
%     match_plot(I1, I2, xy_src, xy_dst,500);

%     match_plot_x(I1,I2,xy_src,xy_dst);
    worldPoints_restored = triangulate(xy_src,...
        xy_dst,...
        cameras(view1).camMatrix, cameras(view2).camMatrix);
    N = N+1;
    res(N).T = time1;
    res(N).view1Name = view1Name;
    res(N).view2Name = view2Name;
    res(N).xy_src = xy_src;
    res(N).mask = I1 > 10/255;
    res(N).xy_dst = xy_dst;
    res(N).view1 = view1;
    res(N).view2 = view2;
    res(N).xyz = worldPoints_restored;
end
% end

%%
function [xy_src,xy_dst] = getSparseMatches(I1,I2)
% I1 = imResample(I1,2);
% I2 = imResample(I2,2);
stackFeats = true;
step = 1;
resampleFactor = 1;
I1 = imResample(I1,resampleFactor,'bilinear');
I2 = imResample(I2,resampleFactor,'bilinear');
[xy_src,xy_dst] = sift_mosaic(I1,I2,I1>10/255,I2>10/255,false,1.5,true,stackFeats,step);
xy_src = xy_src(:,1:2)/resampleFactor;
xy_dst = xy_dst(:,1:2)/resampleFactor;

