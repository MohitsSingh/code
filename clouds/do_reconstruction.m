%%
% todo:
% 1. solve through spacetime, i.e once through diagonal, once through time
% 2. make sure for some world point that it is projected to correct image
% point at several,views, to be sure camera matrix is correct

%% do this one

%% solve view directly
%views = [1 5 9 13 17];
%for vv = 1:4:13
% R = load('Calib_images');
% 
% my_offset = 0;%-1;
% z_offset = 1;
% [world_to_cam_samples,patterns] = generateGroundTruth(A,my_offset,z_offset);
% 

% cameras = struct('camMatrix',cellfun2(@(x) x',{world_to_cam_samples.projMat}));
% ff_mine = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,true,'sparse');
% 
%%
% ff_sanity = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,true,'sanity');
% %
% analyze_errors(ff_mine,ff_sanity,world_to_cam_samples,false);
% 
% world_to_cam_samples_orig = world_to_cam_samples;

% for t = 1:length(world_to_cam_samples)
%     world_to_cam_samples(t).world(:,3) = world_to_cam_samples_orig(t).world(:,3)-1;
% end
%%
for offset = 4
    for startView = 9
        %views = [startView startView+offset startView+2*offset];
        views = [startView, startView+offset, startView+2*offset];% startView+2*offset];
        times = [1 1 1];
%         views = [1 5 9 13 17];
        times_and_views = struct('times',times,'views',views);
        % ff = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,false);
        %[X,Y,Z] = world_to_surface(ff.xyz);
        ff_sanity = getFeatureMatches(times_and_views,B.Img,world_to_cam_samples,cameras,true,'sanity');
      %
      % make dists reflect a similarity measure between matches.
      
        ff_mine = getFeatureMatches(times_and_views,B.Img,world_to_cam_samples,cameras,true,'sparse');        
        ff_mine1 = findConsistentMatches(ff_mine);                        
        analyze_errors(ff_mine1(1),ff_sanity(1),world_to_cam_samples,false);
        analyze_errors(ff_mine(1),ff_sanity(1),world_to_cam_samples,false);
                        
        s1 = ff_mine(1).xy_dst;
        s2 = ff_mine(2).xy_src;
                
        U = l2(ff_sanity1.xyz,world_to_cam_samples(startView).world);
        % S = ff_sanity
        S = ff_mine;   
          % D2_analysis2
        %
        %bb = getSingleRect(true);
        bb = [213  175  251  214];
        figure(1); x2(ff_mine.I1Rect);
        plotBoxes(bb);        
        figure(2)
        ii = inBox(bb,ff_mine.xy_src_rect);                
        match_plot_x(ff_mine.I1Rect,ff_mine.I2Rect,ff_mine.xy_src_rect(ii,:),ff_mine.xy_dst_rect(ii,:))        
        
        match_plot_x(ff_mine(1).I1Rect,ff_mine(1).I2Rect,ff_mine(1).xy_src_rect(:,:),ff_mine(1).xy_dst_rect(:,:))        
          
        x2(ff_mine(1).I1Rect); plotPolygons(ff_mine(1).xy_src_rect,'r.');
        x2(ff_mine(1).I2Rect); plotPolygons(ff_mine(1).xy_dst_rect,'r.');
        %     ff_mine_orig = ff_mine;        
        %     ff_mine = removeOutliers(ff_mine_orig);
        %
        %     showOneByOne=true;
%         close all
        clf;
%         figure('Name',num2str(views),'NumberTitle','off');
%         maximizeFigure;
        analyze_errors(ff_mine,ff_sanity,world_to_cam_samples,true);
%         saveas(gcf,sprintf('results/new_offset_%03.0f_%02.0f_%02.0f.png',offset,views(1),views(2)));
    end
end
%%

if 0 % change this to 1 to see how features can be matched across space and time
    % for now, the resulting reconstruction is quite sparse.
    
    %
    
    % Step 1: match across space-time diagonal
    start = 1;
    skip = 4;
    views = 1:5;
    times = 1:5;
    %when matching features, remove those which blatantly disobey
    %epipolar constraints between the two views ( be lax about it since the
    %clouds change between time steps)
    times_and_views = struct('times',times,'views',views);
    [matches_spacetime_pairs] = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,true);
    
    % now we have for each time/space step a set of matches, propagate those
    % matches to find a matching between time/space = [1,1] to time/space =
    % [5,5]
    
    T = .5;
    [src,dst] = propagate_matches(matches_spacetime_pairs,T);
    
    I1 = imread(fullfile(origDir,matches_spacetime_pairs(1).view1Name));
    I2 = imread(fullfile(origDir,matches_spacetime_pairs(end).view2Name));
    match_plot_x(I1,I2,src,dst,false,50);
    x2(I1); plotPolygons(src,'r.');
    x2(I2); plotPolygons(dst,'r.');
    % now we can constrain the matches, as we know to where each
    
    % now find matches in time: time advances by 4, viewpoint remains the same
    views = [1 1];
    times = [1 5];
    times_and_views = struct('times',times,'views',views);
    matches_onlytime = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,false,'sparse');
    
    % show some matches...
    I1 = imread(fullfile(origDir,matches_onlytime(1).view1Name));
    I2 = imread(fullfile(origDir,matches_onlytime(1).view2Name));
    % % % match_plot_x(I1,I2,matches_onlytime(1).xy_src,matches_onlytime(1).xy_dst,50);
    x2(I1); plotPolygons(matches_onlytime(1).xy_src,'r.')
    x2(I2); plotPolygons(matches_onlytime(1).xy_dst,'r.')
    
    %
    % now we have matches between t_1-1 and t_5-1 and also between t_1-1 and t_5-5. So
    % we can match the points in 5-1 to 5-5 :
    src1 = matches_onlytime.xy_src; % (source in 1,1 for time)
    src2 = src; % (source in 1,1 for diagonal)
    
    dst1 = matches_onlytime.xy_dst;
    dst2 = dst;
    [src_51,dst_55] = propagate_through(dst1,dst2,src1,src2);
    I1 = imread(fullfile(origDir,matches_onlytime(1).view2Name));
    I2 = imread(fullfile(origDir,matches_spacetime_pairs(end).view2Name));
    % match_plot_h(I1,I2,src_51,dst_55,false);
    match_plot_x(I1,I2,src_51,dst_55,false,10);
    
    [src_51_p,dst_55_p] = pruneOutliers(src_51,dst_55);
    match_plot(I1,I2,src_51_p,dst_55_p,50);
    match_plot_x(I1,I2,src_51_p,dst_55_p,true,10);
    
    % reconstruct using tracked matches:
    worldPoints_restored = triangulate(src_51_p,...
        dst_55_p,...
        cameras(1).camMatrix, cameras(5).camMatrix);
    plot3(worldPoints_restored(:,1),worldPoints_restored(:,2),worldPoints_restored(:,3),'r.');
    
end


%%
worldPoints_restored = triangulate(world_to_cam_samples(1).p_target1,...
        world_to_cam_samples(5).p_target1,...
        cameras(1).camMatrix, cameras(5).camMatrix);
num2str(worldPoints_restored)

%%
views = [1 5 9 13 17];
times_and_views = struct('times',ones(size(views)),'views',views);

matches = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,false,'sparse');
save matches.mat matches

%%
load('stereo_res')
i = 1;
%%
R = double(squeeze(res(i,:,:)));
x2(R.*(I1>10/255))

R = 32*R/128;
I1 = ff_sanity.I1Rect;
R = double(R);
x2(I1)

178-186

% (195,178)



%%
[x,y] = meshgrid(1:size(R,1),1:size(R,2));
x_dst = x+R;
y_dst = y;
xy_src = [x(:) y(:)];
goods = inMask(xy_src,(I1>10/255));
xy_src = xy_src(goods,:);
xy_dst = [x_dst(:) y_dst(:)]; xy_dst = xy_dst(goods,:);
xy_src1 = matches(1).tform1.transformPointsInverse(xy_src);
xy_dst1 = matches(1).tform2.transformPointsInverse(xy_dst);
x2(matches(1).I1Rect); plotPolygons(xy_src,'r.')
x2(matches(1).I2Rect); plotPolygons(xy_dst,'r.')

% I1Rect = 


%
%%
I1Rect = ff_mine.I1Rect;
I2Rect = ff_mine.I2Rect;
% I1Rect = I1Rect(100:300,100:300);
% I2Rect = I2Rect(100:300,100:300);
I1Rect = im2uint8(I1Rect);I2Rect = im2uint8(I2Rect);
%%

Z = make_disparity(I1Rect,ff_sanity.xy_src_rect-99,ff_sanity.xy_dst_rect-99);


Z(I1Rect<10)=nan;

S = disparity(I2Rect,I1Rect,'method','SemiGlobal','DisparityRange',[-16 16],'BlockSize',5,'UniquenessThreshold',5);
S(S<-1000) = nan;
mask = I1Rect>10;
S(~mask) = nan;
[src,dst] = disparity_to_src_dst(S);


% x2(I1Rect);
S(I1Rect<10)=nan;
S(S<-1000) = nan;
mask = I1Rect>10;
S(~mask) = nan;
[src,dst] = disparity_to_src_dst(S);

match_plot_x(I1Rect,I2Rect,src,dst,true);

x2(S);
x2(Z);

