
%% solve view directly
%views = [1 5 9 13 17];
for vv = 1:4:13
    views = [vv vv+4];
    times = ones(size(views));
    times_and_views = struct('times',times,'views',views);
    % ff = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,false);
    %[X,Y,Z] = world_to_surface(ff.xyz);
    ff_sanity = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,true,'sanity');
    ff_mine = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,true,'sparse');
    % ff_mine = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,true,'deep');
    % S = ff_sanity
    % S = ff_mine;
    % D2_analysis2
    
    %
    analyze_errors(ff_mine,ff_sanity);
    saveas(gcf,sprintf('results/%05.0f.png',vv))    
end
%% visualize errors
% idists = randperm(length(idists));
for id = 1:length(idists)
    t = idists(id);
    clf;
    s1 = my_src(t,:);
    s2 = gt_dst(t,:);
    s1 = inflatebbox([s1 s1],90,'both',true);
    s2 = inflatebbox([s2 s2],90,'both',true);
    T1 = ff_sanity.I1Rect;
    T2 = ff_sanity.I2Rect;
    %     T1 = ZZ1;
    %     T2 = ZZ2;
    subplot(2,1,1); imagesc2(T1);
    plotPolygons(my_src(t,:),'r.');
    plotPolygons(gt_src(t,:),'g.');
    xlim(s1([1 3]));ylim(s1([2 4])); title('source');
    subplot(2,1,2); imagesc2(T2);
    plotPolygons(my_dst(t,:),'r.');
    plotPolygons(gt_dst(t,:),'g.'); title('target');
    legend({'my match','gt match'});
    xlim(s2([1 3]));ylim(s2([2 4]));
    disp(dists(t))
    pause
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