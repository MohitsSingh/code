% recover the reconstuction

% rmpath('/home/amirro/code/3rdparty/piotr_toolbox/classify/');

function matches = getFeatureMatches(times_and_views,images,world_to_cam_samples,cameras,enforceConstraints,method)
% addpath('/home/amirro/code/3rdparty/SED/');
% load('modelFinal.mat');
global matchedDir
global origDir
global catchSystematicError
transformationsDir = 'xforms';ensuredir(transformationsDir);
imgPattern = 'Image_T_%02.0f_A_%02.0f.png';
warpPattern = fullfile(matchedDir,'T_%02.0f_%02.0f_%02.0f_%02.0f.png');
flowPattern = fullfile(matchedDir,'T_%02.0f_%02.0f_%02.0f_%02.0f_dense.txt');
deepMatchingCmdPattern = '%s/deepmatching-static %s %s -png_settings -improved_settings -subref 1 -out  %s';
matches = struct('T',{},'view1',{},'view2',{},'xyz',{},'uvw',{},'mask',{});
if nargin < 5
    enforceConstraints = true;
end
if nargin < 6
    method = 'sparse';
end
userOrigImagesToo = false;
prune_heuristic = 15;
N = 0;
dists = 0;
xy_src_rect = [];
xy_dst_rect=  [];

for it = 1:length(times_and_views.times)-1
    it
    N = N+1;
    % obtain image data
    time1 = times_and_views.times(it);
    time2 = times_and_views.times(it+1);
    view1 = times_and_views.views(it);
    view2 = times_and_views.views(it+1);
    view1Name = sprintf(imgPattern,time1,view1);
    view2Name = sprintf(imgPattern,time2,view2);
    view1Path = fullfile(origDir,view1Name);
    view2Path = fullfile(origDir,view2Name);
    I1 = im2single(images{time1,view1});
    I2 = im2single(images{time2,view2});
%     I1 = im2single(imread(view1Path));
%     I2 = im2single(imread(view2Path));
    pts1 = [];
    pts2 = [];
    % rectify images
    if view1~=view2
        [I1Rect,I2Rect,tform1,tform2,pts1,pts2] = rectify_helper(I1,I2,world_to_cam_samples(view1),...
            world_to_cam_samples(view2));
        
        %         return;
        %
        %         [I1Rect,I2Rect,tform1,tform2] = rectify_helper2(I1,I2);
        
    else
        I1Rect = I1;
        I2Rect = I2;
    end
    
    matches(N).T = time1;
    matches(N).I1 = I1;
    matches(N).I2 = I2;
    matches(N).I1Rect = I1Rect;
    matches(N).I2Rect = I2Rect;
    matches(N).view1Name = view1Name;
    matches(N).view2Name = view2Name;
    matches(N).tform1=tform1;matches(N).tform2=tform2;
    %     warning('skipping matching');
    %     continue;
    
    mask = (I1Rect>10/255);
    zzz = enforceConstraints && time1==time2 && view1~=view2;
    %
    
    if strcmp(method,'sanity')
        w1 = world_to_cam_samples(view1);
        w2 = world_to_cam_samples(view2);
        
        pts1_w = w1.world;
        pts2_w = w2.world;
        [c,ia,ib] = intersect(pts1_w,pts2_w,'rows');
        if isempty(pts1)
            pts1 = w1.cam;
            pts2 = w2.cam;
            pts1 = pts1(ia,:);
            pts2 = pts2(ib,:);
        end
        xy_src = pts1;
        xy_dst = pts2;
        xy_src_rect = tform1.transformPointsForward(xy_src);
        xy_dst_rect = tform2.transformPointsForward(xy_dst);
        
    else
        %
        if strcmp(method,'sparse')
            [xy_src,xy_dst,dists] = getSparseMatches2(I1Rect,I2Rect,1,false);
        elseif strcmp(method,'dense')
            %%
            for blockSize = 11
            
                %S = disparity(I2Rect,I1Rect,'method','SemiGlobal','DisparityRange',[-16 16],'BlockSize',11,'UniquenessThreshold',5);
                S = disparity(I2Rect,I1Rect,'method','SemiGlobal','DisparityRange',[-16 16],'BlockSize',blockSize,...
                    'UniquenessThreshold',45,...
                    'ContrastThreshold',.5,'DistanceThreshold',1);
%                 S = disparity(I2Rect,I1Rect,'method','BlockMatching','DisparityRange',[-16 16],'BlockSize',blockSize,'UniquenessThreshold',5);
                S(S<-1000) = nan;
                mask = I1Rect>10/255;
                S(~mask) = nan;
                figure; imagesc2(S);
%                 dpc
            end
            
            %%
            [xy_src,xy_dst] = disparity_to_src_dst(S);
            
            
        elseif strcmp(method,'dsp')
            [xy_src,xy_dst] = getDSPMatches(I1Rect,I2Rect,1,true);
        elseif strcmp(method,'deep')
            im1Name = sprintf(warpPattern,time1,view1,time2,view2)
            im2Name = sprintf(warpPattern,time2,view2,time1,view1)
            imwrite(I1Rect,im1Name);
            imwrite(I2Rect,im2Name);
            path_to_deep_matching = '~/code/3rdparty/deepmatching_1.0.2_c++';
            matchFile = sprintf(flowPattern,time1,view1,time2,view2);
            if ~exist(matchFile,'file')
                deepMatchingCmd = sprintf(deepMatchingCmdPattern,path_to_deep_matching,im1Name,im2Name,matchFile);
                [status,result] = system(deepMatchingCmd);
            end
            R = dlmread(matchFile);
            [z,iz] = sort(R(:,5),'descend');
            xy_src = R(iz,1:2);
            xy_dst = R(iz,3:4);
            g = inMask(xy_src,I1Rect>10/255);
            xy_src = xy_src(g,:);
            xy_dst = xy_dst(g,:);
            xy_src_rect = xy_src;
            xy_dst_rect = xy_dst;
            
        else
            [xy_src,xy_dst] = getSiftMatches(I1Rect,I2Rect);
        end
        if zzz
            yDiff = abs(xy_src(:,2)-xy_dst(:,2));
            %             figure(1); clf; hist(yDiff,0:50);drawnow;
            %             return
            %             exit
            if (catchSystematicError && 0)
                warning('------------>NOTICE: experimentally taking mode of y disparities to be true');
                [m,f] = mode(yDiff);
                if f/length(yDiff)>.3
                    yDiff = yDiff-m;
                end
            end
            goods = yDiff<=1;
            xy_src = xy_src(goods,:);
            xy_dst = xy_dst(goods,:);
            if any(dists)
                dists = dists(goods,:,:);
            end
        end
        
        if userOrigImagesToo
            [xy_orig_src,xy_orig_dst] = getDSPMatches(I1,I2,1,false);
        end
        %     [xy_src,xy_dst] = getDeepMatches(I1_,I2d_,view1,view2,time1,time2,origDir, imgPattern,flowPattern,model)
        % remove matches which do not conform to the epipolar geometry
        if view1~=view2
            xy_src_rect = xy_src;
            xy_dst_rect = xy_dst;
            xy_src = tform1.transformPointsInverse(xy_src);
            xy_dst = tform2.transformPointsInverse(xy_dst);
        end
    end
    
    if enforceConstraints
        if userOrigImagesToo
            [xy_orig_src,xy_orig_dst,z] = pruneOutliers(xy_orig_src,xy_orig_dst,[],0);
        end
    end
    
    if (0)
        xy_src = tform1.transformPointsForward(xy_src);
        xy_dst = tform2.transformPointsForward(xy_dst);
        
        [norms,norm_diffs] = computeDisparityOutliers(xy_src,xy_dst);
        
        [r,ir] = sort(norm_diffs,'descend');
        match_plot_x(I1_,I2_,xy_src(ir,:),xy_dst(ir,:),1);
    end
    
    
    
    if userOrigImagesToo
        xy_src = [xy_src;xy_orig_src];
        xy_dst = [xy_dst;xy_orig_dst];
    end
    %
    
    
    
    matches(N).xy_src_rect = xy_src_rect;
    matches(N).xy_dst_rect = xy_dst_rect;
    matches(N).xy_src = xy_src;
    matches(N).mask = mask;
    matches(N).xy_dst = xy_dst;
    matches(N).view1 = view1;
    matches(N).view2 = view2;
    matches(N).dists = dists;
    
    if view1~=view2 && time1==time2 % recover structure, unless same view or differnt time
        worldPoints_restored = triangulate(xy_src,...
            xy_dst,...
            cameras(view1).camMatrix, cameras(view2).camMatrix);
        matches(N).xyz = worldPoints_restored;
    end
end