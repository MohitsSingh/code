function [I1Rect,I2Rect,tform1,tform2,pts1,pts2] = rectify_helper(I1,I2,w1,w2)
if isstruct(w1)
    pts1 = w1.cam;
    pts2 = w2.cam;
    pts1_w = w1.world;
    pts2_w = w2.world;
    [c,ia,ib] = intersect(pts1_w,pts2_w,'rows');
    
    pts1 = pts1(ia,:);
    pts2 = pts2(ib,:);
else
    pts1 = w1;
    pts2 = w2;
end

[I1Rect,I2Rect,tform1,tform2] = estimateF_Helper(pts1,pts2,I1,I2);

%

% find if there's a systematic error and correct it:
global catchSystematicError
% % % if (catchSystematicError)
% % %     [xy_src,xy_dst] = getSiftMatches(I1Rect,I2Rect);
% % %     
% % %     % [means, covs, p] = vl_gmm(xy_src'-xy_dst', 2);
% % %     y_diff = abs(xy_src(:,2)-xy_dst(:,2));
% % %     
% % %     [no1,xo1] = hist(y_diff,-16:1:16);
% % %     [z,iz] = max(no1);
% % %     y = xo1(iz);
% % %     
% % %     % throw away all values outside this range and try again
% % %     y_diff(abs(y_diff-y)>.5) = [];
% % %     [no,xo] = hist(y_diff,y-.5:.1:y+.5);
% % %     [z,iz] = max(no);
% % %     y = xo(iz);
% % %     
% % %     % figure(1); clf;
% % %     % subplot(1,2,1); bar(xo1,no1);
% % %     % subplot(1,2,2);subplot(1,2,1); bar(xo,no);
% % %     
% % %     
% % %     disp(['found a systematic error of ' num2str(y) ' pixels']);
% % %     pts2_orig =pts2;
% % %     pts2 = tform2.transformPointsForward(pts2)+y;
% % %     pts2 = tform2.transformPointsInverse(pts2);
% % %     
% % %     
% % %     
% % %     % disp(['in original coordinates, this amounts to a difference
% % %     %
% % %     I1Rect_ = I1Rect;
% % %     I2Rect_ = I2Rect;
% % %     [I1Rect,I2Rect,tform1,tform2] = estimateF_Helper(pts1,pts2,I1,I2);
% % %     
% % % end


if (catchSystematicError)
    [xy_src,xy_dst] = getSiftMatches(I1Rect,I2Rect);
    
    % [means, covs, p] = vl_gmm(xy_src'-xy_dst', 2);
    y_diff = abs(xy_src(:,2)-xy_dst(:,2));
    
    xy_src(y_diff > 2,:) = [];
    xy_dst(y_diff > 2,:) = [];
    y_diff(y_diff > 2) = [];
    % throw away all values outside this range and try again
%     y_diff(abs(y_diff-y)>.5) = [];
    y = mean(y_diff);
    %[no,xo] = hist(y_diff
%     [z,iz] = max(no);
%     y = xo(iz);
    
    % figure(1); clf;
    % subplot(1,2,1); bar(xo1,no1);
    % subplot(1,2,2);subplot(1,2,1); bar(xo,no);
    
    
    disp(['found a systematic error of ' num2str(y) ' pixels']);
    pts2_orig =pts2;
    pts2 = tform2.transformPointsForward(pts2)+y;
    pts2 = tform2.transformPointsInverse(pts2);
    
    
    
    % disp(['in original coordinates, this amounts to a difference
    %
    I1Rect_ = I1Rect;
    I2Rect_ = I2Rect;
    [I1Rect,I2Rect,tform1,tform2] = estimateF_Helper(pts1,pts2,I1,I2);
    
end

if 0
    x2(I1Rect); plotPolygons(tform1.transformPointsForward(pts1),'r.');
    x2(I2Rect); plotPolygons(tform2.transformPointsForward(pts2),'r.');
    match_plot_x(I1Rect,I2Rect,tform1.transformPointsForward(pts1),tform2.transformPointsForward(pts2),true);
end

function [I1Rect,I2Rect,tform1,tform2] = estimateF_Helper(pts1,pts2,I1,I2,nTrials,method,confidence)
if nargin < 5
    nTrials = 10000;
end
if nargin < 6
    method = 'RANSAC';
end
if nargin < 7
    confidence = 99;
end
[fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
    pts1, pts2, 'Method', 'RANSAC', ...
    'NumTrials', 10000, 'DistanceThreshold', 0.1, 'Confidence', confidence);
inlierPoints1 = pts1(epipolarInliers, :);
inlierPoints2 = pts2(epipolarInliers, :);
[t1, t2] = estimateUncalibratedRectification(fMatrix, ...
    inlierPoints1, inlierPoints2, size(I2));
tform1 = projective2d(t1);
tform2 = projective2d(t2);
I1Rect = imwarp(I1, tform1, 'OutputView', imref2d(size(I1)));
I2Rect = imwarp(I2, tform2, 'OutputView', imref2d(size(I2)));


