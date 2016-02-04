% recover the reconstuction

% rmpath('/home/amirro/code/3rdparty/piotr_toolbox/classify/');

function matches = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,enforceConstraints)
% addpath('/home/amirro/code/3rdparty/SED/');
% load('modelFinal.mat');


matchedDir = '/home/amirro/code/clouds/matched_pairs';
origDir = '/home/amirro/code/clouds/Images_divided_by_maxValue';
transformationsDir = 'xforms';ensuredir(transformationsDir);
imgPattern = 'Image_T_%02.0f_A_%02.0f.png';
warpPattern = fullfile(matchedDir,'T_%02.0f_%02.0f_%02.0f.png');
flowPattern = fullfile(matchedDir,'T_%02.0f_%02.0f_%02.0f_%02.0f.mat');
matches = struct('T',{},'view1',{},'view2',{},'xyz',{},'uvw',{},'mask',{});
if nargin < 4
    enforceConstraints = true;
end
userOrigImagesToo = false;
prune_heuristic = 15;
N = 0;
for it = 1:length(times_and_views.times)-1
    it
    % obtain image data
    time1 = times_and_views.times(it);
    time2 = times_and_views.times(it+1);
    view1 = times_and_views.views(it);
    view2 = times_and_views.views(it+1);
    view1Name = sprintf(imgPattern,time1,view1);
    view2Name = sprintf(imgPattern,time2,view2);
    view1Path = fullfile(origDir,view1Name);
    view2Path = fullfile(origDir,view2Name);
    I1 = im2single(imread(view1Path));
    I2 = im2single(imread(view2Path));
    
    % rectify images
    if view1~=view2
        [I1Rect,I2Rect,tform1,tform2] = rectify_helper(I1,I2,world_to_cam_samples(view1),...
            world_to_cam_samples(view2));
    else
        I1Rect=I1;
        I2Rect=I2;
    end
    
    mask = (I1Rect>10/255);
    [xy_src,xy_dst] = getDSPMatches(I1Rect,I2Rect,1);
    
    if userOrigImagesToo
        [xy_orig_src,xy_orig_dst] = getSparseMatches2(I1,I2,1);
    end
    %     [xy_src,xy_dst] = getDeepMatches(I1_,I2_,view1,view2,time1,time2,origDir, imgPattern,flowPattern,model)
    % remove matches which do not conform to the epipolar geometry
    if view1~=view2
        xy_src = tform1.transformPointsInverse(xy_src);
        xy_dst = tform2.transformPointsInverse(xy_dst);
    end
    
    %     II=  zeros([size2(I1) 3]);
    %     II(:,:,1) = I1;
    %     II(:,:,2) = I2;
    %     clf; imagesc2(II);
    %     norms= vec_norms(xy_src-xy_dst);
    %     sel_ = norms<20;
    %     quiver2(xy_src(sel_,:),xy_dst(sel_,:)-xy_src(sel_,:),0,'m');
    %     x2(I1);x2(I2);
    if enforceConstraints
        
        [xy_src,xy_dst,z] = pruneOutliers(xy_src,xy_dst,[],0);
        if userOrigImagesToo
            [xy_orig_src,xy_orig_dst,z] = pruneOutliers(xy_orig_src,xy_orig_dst,[],0);
        end
        %
        %         [~,~,z1] = pruneOutliers(xy_src,xy_dst,fMatrix);
        %         [~,~,z2] = pruneOutliers(xy_orig_src,xy_orig_dst,fMatrix);
        %
        %         y_diff = abs(xy_src(:,2)-xy_dst(:,2));
        %         T = 2;
        %         xy_src = xy_src(y_diff<=T,:);
        %         xy_dst = xy_dst(y_diff<=T,:);
    end
    
    if (0)
        xy_src = tform1.transformPointsForward(xy_src);
        xy_dst = tform2.transformPointsForward(xy_dst);
        
        [norms,norm_diffs] = computeDisparityOutliers(xy_src,xy_dst);
        %         vec_norms = @(x) sum(x.^2,2).^.5;
        
        [r,ir] = sort(norm_diffs,'descend');
        match_plot_x(I1_,I2_,xy_src(ir,:),xy_dst(ir,:),1);
        %         match_plot_x(I1_,I2_,xy_src,xy_dst,50);
    end
    
    
    %     [r,ir] = sort(abs(z1),'ascend');
    %     for it = 1:100:length(ir)
    %         it
    %         match_plot_x(I1_,I2_,xy_src(ir(it),:),xy_dst(ir(it),:),1);
    %         title(num2str(r(it)));
    %         dpc;
    %
    %     end
    
    
    if userOrigImagesToo
        xy_src = [xy_src;xy_orig_src];
        xy_dst = [xy_dst;xy_orig_dst];
    end
    %
    
    N = N+1;
    matches(N).T = time1;
    matches(N).view1Name = view1Name;
    matches(N).view2Name = view2Name;
    matches(N).xy_src = xy_src;
    matches(N).mask = mask;
    matches(N).xy_dst = xy_dst;
    matches(N).view1 = view1;
    matches(N).view2 = view2;
    
    if view1~=view2 && time1==time2 % recover structure, unless same view or differnt time
        worldPoints_restored = triangulate(xy_src,...
            xy_dst,...
            cameras(view1).camMatrix, cameras(view2).camMatrix);
        matches(N).xyz = worldPoints_restored;
    end
    
    
end
% end

%


function [xy_src,xy_dst] = getDeepMatches(I1,I2,view1,view2,time1,time2,imgsDir,imgPattern,flowPattern,model)
path_to_deep_matching = '~/code/3rdparty/deepmatching_1.0.2_c++';
options = '';
path_to_epic_flow = '/home/amirro/code/3rdparty/EpicFlow_v1.00';
M = 'matched_pairs';
fileNames = dir(fullfile(M,'*.png'));
% flows = struct('img1Name',{},'flow',{});

% for each file, look for a file with the same name but with t+1
N = 0;

% fileNames = dir(fullfile(M,'T_*_01_05.png'));

% for t = 1:length(fileNames)
% t
view1Name = sprintf(imgPattern,time1,view1);
view2Name = sprintf(imgPattern,time2,view2);
view1Path = fullfile(imgsDir,['w_' view1Name view2Name]);
view2Path = fullfile(imgsDir,['w_' view2Name view1Name]);
imwrite(I1,view1Path);
imwrite(I2,view2Path);
I = I1;
if size(I,3)==1, I = cat(3,I,I,I); end;
edgesFile = fullfile('edges',[view1Name '_' view2Name '.edges']);
if ~exist(edgesFile,'file')
    edges = edgesDetect(I, model);
    fid=fopen(edgesFile,'wb');
    fwrite(fid,transpose(edges),'single');
    fclose(fid);
end
matchFile = [sprintf(flowPattern,time1,view1,time2,view2) '.match.txt'];
if (~exist(matchFile,'file'))
    deepMatchingCmd = sprintf('%s/deepmatching-static %s %s -png_settings -improved_settings -downscale 0 -out  %s',...
        path_to_deep_matching,view1Path,view2Path,matchFile);
    [status,result] = system(deepMatchingCmd);
end

%     end

flow_out_file = [strrep(matchFile,'matched_pairs','flows') '.flo'];
if ~exist(flow_out_file,'file')
    epicflowCmd = sprintf('~/code/3rdparty/EpicFlow_v1.00/epicflow-static %s %s %s %s %s %s -a .5 -d 1',...
        view1Path,view2Path,edgesFile,matchFile,flow_out_file,options);
    [status,result] = system(epicflowCmd);
end

Z = dlmread(matchFile);
xy_src=Z(:,1:2)+1;
xy_dst=Z(:,3:4)+1;

match_plot_x(I1,I2,xy_src,xy_dst,true)

%     end
N = N+1;
img1Name = im1Name;
img2Name = im2Name;
flow = readFlowFile(flow_out_file);
flowResPath = fullfile('flows',[view1Name '.mat']);
% for t = 1:length(flows)
%     flow = flows(t).flow;
if ~isempty(flow)
    u = squeeze(flow(:,:,1));
    v = squeeze(flow(:,:,2));
    save(flowResPath,'u','v');
end


function [xy_src,xy_dst] = getSparseMatches2(I1,I2,resampleFactor)
I1 = imResample(I1,resampleFactor);
I2 = imResample(I2,resampleFactor);
m1 = I1>10/255;
m2 = I2>10/255;
bounds1 = region2Box(m1);
bounds1(1:2) = bounds1(1:2)-15;
bounds1(3:4) = bounds1(3:4)+15;
bounds2 = region2Box(m2);
bounds2(1:2) = bounds2(1:2)-15;
bounds2(3:4) = bounds2(3:4)+15;
[f1,d1] = vl_dsift(I1,'Step',1,'Bounds',bounds1);
[f2,d2] = vl_dsift(I2,'Step',1,'Bounds',bounds2);
f1_inds = sub2ind2(size2(I1),round(f1([2 1],:)'));
f1_sel = m1(f1_inds);
f1 = f1(:,f1_sel);
d1 = d1(:,f1_sel);
f2_inds = sub2ind2(size2(I2),round(f2([2 1],:)'));
f2_sel = m2(f2_inds);
f2 = f2(:,f2_sel);
d2 = d2(:,f2_sel);
% matches = best_bodies_match(d1',d2');
% %             matches = knn_match(d1',d2');
% xy_src = f1(1:2,matches(1,:))'/resampleFactor;
% xy_dst = f2(1:2,matches(2,:))'/resampleFactor;

xy_src_ = {};
xy_dst_ = {};
for t = 1:1
    t
    
    matches = best_bodies_match(d1',d2');
    
    z1 = false(1,size(d1,2));
    z2 = false(1,size(d2,2));
    z1(matches(1,:)) = true;
    z2(matches(2,:)) = true;
    %             matches = knn_match(d1',d2');
    xy_src_{end+1}= f1(1:2,matches(1,:))'/resampleFactor;
    xy_dst_{end+1}= f2(1:2,matches(2,:))'/resampleFactor;
    
    f1 = f1(:,~z1);
    f2 = f2(:,~z1);
    d1 = d1(:,~z1);
    d2 = d2(:,~z1);
end

xy_src = cat(1,xy_src_{:});
xy_dst = cat(1,xy_dst_{:});

% % now, make this more robust by searching only in allowed neighborhoods
% diffs = vec_norms(xy_dst-xy_src);
% dists_xy = l2(f1',f2').^.5;
% matches = best_bodies_match(d1',d2',dists_xy<mean(diffs)+std(diffs));

%
function [xy_src,xy_dst] = getSparseMatches(I1,I2,resampleFactor)

% I1 = imResample(I1,2);
% I2 = imResample(I2,2);
I1 = imResample(I1,resampleFactor);
I2 = imResample(I2,resampleFactor);

stackFeats = true;
step = 1;
resampleFactor = 1;
I1 = imResample(I1,resampleFactor,'bilinear');
I2 = imResample(I2,resampleFactor,'bilinear');
[xy_src,xy_dst] = sift_mosaic(I1,I2,I1>10/255,I2>10/255,false,1.5,true,stackFeats,step);
xy_src = xy_src(:,1:2)/resampleFactor;
xy_dst = xy_dst(:,1:2)/resampleFactor;

function [xy_src,xy_dst] = getDSPMatches(I1,I2,resampleFactor)
I1 = imResample(I1,resampleFactor,'bilinear');
I2 = imResample(I2,resampleFactor,'bilinear');

I1_ = I1;
I2_ = I2;
pca_basis = [];
sift_size = 4;

m1 = I1>10/255;
m2 = I2>10/255;
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
[vx,vy] = DSPMatch(sift1, sift2);
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

xy_src = bsxfun(@plus,xy_src,bounds1(1:2)+bbox1([1 3])')-2;
xy_dst = bsxfun(@plus,xy_dst,bounds1(1:2)+bbox2([1 3])')-2;

% x2(I1_); plotPolygons(xy_src,'r.')
% x2(I2_); plotPolygons(xy_dst,'r.')
% match_plot_x(I1_,I2_,xy_src,xy_dst,25);

% Match
%tic;
% xy_dst = xy_src + [vx(:) vy(:)];

function [I1Rect,I2Rect,tform1,tform2] = rectify_helper(I1,I2,w1,w2)
pts1 = w1.cam;
pts2 = w2.cam;
w2.cam;
useRotation=false;
if ~useRotation
    [fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
        pts1, pts2, 'Method', 'RANSAC', ...
        'NumTrials', 10000, 'DistanceThreshold', 0.1, 'Confidence', 99.99);
    inlierPoints1 = pts1(epipolarInliers, :);
    inlierPoints2 = pts2(epipolarInliers, :);
    [t1, t2] = estimateUncalibratedRectification(fMatrix, ...
        inlierPoints1, inlierPoints2, size(I2));
    tform1 = projective2d(t1);
    tform2 = projective2d(t2);
    I1Rect = imwarp(I1, tform1, 'OutputView', imref2d(size(I1)));
    I2Rect = imwarp(I2, tform2, 'OutputView', imref2d(size(I2)));
else
    T = fitgeotrans(pts1,pts2,'nonreflectivesimilarity');
    tform1 = projective2d(T.T);
    tform2 = projective2d(eye(3));
    I1Rect = imwarp(I1, tform1, 'OutputView', imref2d(size(I1)));
    I2Rect = imwarp(I2, tform2, 'OutputView', imref2d(size(I2)));
end