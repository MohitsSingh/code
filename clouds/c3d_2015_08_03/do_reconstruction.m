
imgDir = '/home/amirro/code/clouds/Images_divided_by_maxValue';
addpath(genpath('/home/amirro/code/3rdparty/imrender/'));
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
imgDir = 'Images_divided_by_maxValue';
I1 = imread(fullfile(imgDir,matches_spacetime_pairs(1).view1Name));
I2 = imread(fullfile(imgDir,matches_spacetime_pairs(end).view2Name));
match_plot_x(I1,I2,src,dst,false,50);
x2(I1); plotPolygons(src,'r.');
x2(I2); plotPolygons(dst,'r.');
% now we can constrain the matches, as we know to where each 

% now find matches in time: time advances by 4, viewpoint remains the same
views = [1 1];
times = [1 5];
times_and_views = struct('times',times,'views',views);
matches_onlytime = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,false);

% show some matches...
I1 = imread(fullfile(imgDir,matches_onlytime(1).view1Name));
I2 = imread(fullfile(imgDir,matches_onlytime(1).view2Name));
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
I1 = imread(fullfile(imgDir,matches_onlytime(1).view2Name));
I2 = imread(fullfile(imgDir,matches_spacetime_pairs(end).view2Name));
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



% [X,Y,Z] = world_to_surface(worldPoints_restored);
% x2(Z)

% now we can reconstruct these matches reliably - by limiting the disparity
% maps at all locations...


%% solve view directly
%views = [1 5 9 13 17];
views = [1 5];
times = ones(size(views));
times_and_views = struct('times',times,'views',views);
% ff = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,false);
%[X,Y,Z] = world_to_surface(ff.xyz);
ff = getFeatureMatches(times_and_views,world_to_cam_samples,cameras,true);
% D2_analysis2

% [norms,norm_diffs] = computeDisparityOutliers(ff.xy_src,ff.xy_dst);
% nd = fitdist(norms,'normal');
% y = pdf(nd,norms);
% 
% [X,Y,Z] = world_to_surface(ff.xyz);%(y>.01,:));
% imagesc(Z); colorbar
% xlim([110 310]);
% ylim([110 310]);
% 
% % compute a transformation to constrain the matches.
% tform = fitgeotrans(ff.xy_src,ff.xy_dst,'projective');
% I1Rect = imwarp(I1, tform, 'OutputView', imref2d(size(I1)));
% x2(I1Rect);
% x2(I2);
% 
% tform = fitgeotrans(ff.xy_dst,ff.xy_src,'projective');
% I2Rect = imwarp(I2, tform, 'OutputView', imref2d(size(I1)));
% x2(I2Rect);
% x2(I1);
% 
% % remove outliers, by finding abnormal disparities
% % for each point, find the distribution of its nearest neighbor's movement
% % and remove it if it doen't make sense.
% 
% [r,ir] = sort(norm_diffs,'descend');
% match_plot_x(I1,I2,ff.xy_src(ir,:),ff.xy_dst(ir,:),1);
% 
% I1 = imread(fullfile(imgDir,ff(1).view1Name));
% I2 = imread(fullfile(imgDir,ff(end).view2Name));
% match_plot_x(I1,I2,ff.xy_src,ff.xy_dst,50);
% 
% plotPolygons(ff.xy_src-ff.xy_dst,'r.')
% 
% x2(I1); plotPolygons(ff.xy_src,'r.')
% x2(I2); plotPolygons(ff.xy_dst,'r.')
% ff = calc_flow(ff);
% ff = make_topo(ff);
% 
% nd = fitdist(n,'normal');
% x_values = min(n):max(n);
% y = pdf(nd,x_values);
% plot(x_values,y)
% y_ = pdf(nd,n);
% plot(n,log(y_),'r.')
% 
% x2(I1); plotPolygons(ff.xy_src(y_<.003,:),'r.')
% x2(I2); plotPolygons(ff.xy_dst(y_<.003,:),'r.')
% plot(sort(ff.xyz(y_<.003,3)))
% 
% [X,Y,Z] = world_to_surface(ff.xyz);
% 
% x2(I1);
% plotPolygons(ff.xy_src,'r.')
% % cameras(1).camMatrix'*ff.xyz
% 
% % further filter by some more measures:
% %back project to image plane
% XY = [ff.xyz ones(size(ff.xyz,1),1)]*cameras(1).camMatrix;
% XY = XY(:,1:2)./XY(:,[3 3]);
% [~,norms] = normalize_vec(XY'-ff.xy_src');
% norms = norms(:);
% % 1. remove points with large reprojection error
% bads = norms > .5;
% 
% mask = I1>10;
% mask = imerode(mask,ones(5));
% inds = sub2ind2(size(I1),round(ff.xy_src(:,[2 1])));
% % bads = bads | mask(inds)==0;
% % x2(I1); plotPolygons(XY(bads,:),'r.')
% [X,Y,Z] = world_to_surface(ff.xyz(~bads,:));
% meshz(X,Y,Z);
% 
% [X,Y,Z] = world_to_surface(ff.xyz);
% meshz(X,Y,Z);
% 
% clf; imagesc2(I1); plotPolygons(XY,'g+');
% plotPolygons(ff.xy_src,'r.')
% plot(sort(norms));
% 
% % x2(Z.*(I1>10/255))
% 
% %%
% 
% all_results = struct('times',{},'views',{},'surfaces',{});
% N = 0;
% % for t = 1
% %     for iView = 1:13
% % only time...
% %times = ones(1,13);
% views = 1:4:17;
% times = ones(size(views));
% % times = 1:5;
% % views = ones(1,5);
% only_space = [times;views]';
% only_space = only_space(1:2,:);
% N = N+1;
% all_results(N).surfaces = recover_structure_new(only_space,world_to_cam_samples,cameras,'sparse');
% % D2_analysis2;
% 
% 
% 
% 
% 
% all_results = struct('times',{},'views',{},'surfaces',{});
% N = 0;
% % for t = 1
% %     for iView = 1:13
% % only time...
% times = 1:13;
% views = ones(1,13);
% % times = 1:5;
% % views = ones(1,5);
% only_time = [times;views]';
% only_time = only_time(1,:);
% N = N+1;
% all_results(N).surfaces = recover_structure(only_time,world_to_cam_samples,cameras,1,'sparse');
% % D2_analysis2;
% %%
% all_results(N).surfaces = calc_flow(all_results(N).surfaces);
% all_results(N).surfaces = make_topo(all_results(N).surfaces);
% 
% all_results(N).times = times;
% all_results(N).views = views;
% all_results(N).name = 'only time';
% 
% % viewStuff(all_results(1).surfaces)
% 
% %     end
% % end
% 
% % space and time...
% times = 1:13;
% views = 1:13;
% time_and_space = [times;views]';
% N = N+1;
% 
% 
% time_and_space = [1 1;2 2];
% 
% all_results(N).surfaces = recover_structure_new(time_and_space,world_to_cam_samples,cameras,'sparse');
% all_results(N).surfaces = calc_flow(all_results(N).surfaces);
% all_results(N).surfaces = make_topo(all_results(N).surfaces);
% 
% all_results(N).times = times;
% all_results(N).views = views;
% all_results(N).name = 'time and space';
% % all_results22222.
% 
% % views = views;
% 
% % save all_results_sparse.mat all_results
% 
% % viewStuff(all_results(1).surfaces)
% % viewStuff(all_results(2).surfaces)
% % save all_results.mat all_results
% % % % 
% % % % save all_results_July_22_2015.mat all_results
% % % % 
% % displayImageSeries2({all_results(2).surfaces.topo})
% 
% % addpath('Validation');
% edit D2_analysis2
% 
% II = zeros([size2(I1) 3]); II(:,:,1)=I1;II(:,:,2)=I2; x2(II);
