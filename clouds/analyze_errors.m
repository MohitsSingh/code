function analyze_errors(ff_mine,ff_sanity,world_to_cam_samples,showOneByOne)
if nargin < 4
    showOneByOne=false;
end
figure(1);clf;
%UNTITLED Summary of this function goes here
%  Detailed explanation goes here

%% analyze errors : for each computed match, find nearest neighbor of source and corresp. destination
% ggg = ff_mine.dists<30000;

n_rows = 2;
n_cols = 3;

my_src = ff_mine.xy_src_rect;
my_dst = ff_mine.xy_dst_rect;
gt_src = ff_sanity.xy_src_rect;
gt_dst = ff_sanity.xy_dst_rect;
[mm,im] = sort(l2(my_src,gt_src).^.5,2,'ascend');
m = mm(:,1);
im = im(:,1);
goodPoints = true(size(im));
% goodPoints = m < .5;
goodPoints_dst = im(goodPoints);

%%
% for the remaining points, find the error of the 3D reconstruction
my_xyz = ff_mine.xyz(goodPoints,:);
my_xyz(:,3) = my_xyz(:,3)+30;
% figure,hist(ff_mine.dists,50)
gt_xyz = ff_sanity.xyz(goodPoints_dst,:);

% d = min(l2(my_xyz,gt_xyz).^.5,[],2);
% goodPoints_dst = goodPoints_dst(:,1);

% plot3(ff_sanity.xyz(:,1),ff_sanity.xyz(:,2),ff_sanity.xyz(:,3),'g.');
% plot3(ff_sanity.xyz(:,1),ff_sanity.xyz(:,2),ff_sanity.xyz(:,3),'g.');

world_center = world_to_cam_samples(ff_mine(1).T).world_center;
xyz_cam = world_to_cam_samples(ff_mine(1).T).xyz_cam;

% R = (xyz_cam-world_center);
% R = 30*R/norm(R);
% my_xyz = bsxfun(@plus,my_xyz,R);

wc = xyz_cam-world_center;
wc = 1000*(wc/norm(wc));
subplot(n_rows,n_cols,1); hold on;
plot3(my_xyz(:,1),my_xyz(:,2),my_xyz(:,3),'r.');

% plot3(world_center(1),world_center(2),world_center(3));
% quiver3(world_center(1),world_center(2),world_center(3),wc(1),wc(2),wc(3),0,'g','LineWidth',2);
% 
% wc = wc/norm(wc);
% figure,plot(sort(my_xyz*wc'))
plot3(gt_xyz(:,1),gt_xyz(:,2),gt_xyz(:,3),'g.');

diff_3 = gt_xyz-my_xyz;
quiver3(my_xyz(:,1),my_xyz(:,2),my_xyz(:,3),diff_3(:,1),diff_3(:,2),diff_3(:,3),0);
title({'3D Reconstruction vs.','ground truth points'});
legend({'recon.','gt'});

[n,x] = hist(diff_3(:,3),-300:15:300);
subplot(n_rows,n_cols,2),bar(x,n);title('Error Distribution(world)');
xlabel('Height Error(meters)'); ylabel('count');grid on; grid minor;
% interpolate to x,y disparities

fprintf('number of total matches: %d\n',length(m));
fprintf('number of sources near ground-truth: %d\n', nnz(goodPoints));
my_src = my_src(goodPoints,:);
my_dst = my_dst(goodPoints,:);
gt_src = gt_src(goodPoints_dst,:);
gt_dst = gt_dst(goodPoints_dst,:);

%assert(all(vec_norms(my_src-gt_src) < 1));

% assert(all(vec_norms(my_src-gt_src) < 1));

%dists = vec_norms(my_dst-gt_dst);

%dists = vec_norms(my_dst-gt_dst);
dists = gt_dst(:,1)-my_dst(:,1);

d1 = ff_mine.dists(:,:,1);
d2 = ff_mine.dists(:,:,2);

D = sum(d1.*d2,2);
% figure,plot(dists,1-D,'r.')

[dists_to_gt,idists] = sort(dists,'descend');
subplot(n_rows,n_cols,3); 
hist(dists_to_gt,(-300:15:300)/30); title('Error Distribution (pixels)');
xlabel('Disparity Error(pixels)'); ylabel('count');grid on; grid minor;

I1Rect = ff_mine.I1Rect;
I1Rect_ = ff_sanity.I1Rect;
assert(all(I1Rect(:)==I1Rect_(:)));

Z_mine = make_disparity(I1Rect,ff_mine.xy_src(goodPoints,:),ff_mine.xy_dst(goodPoints,:));
mask = I1Rect>10/255;

% x2(Z_mine.*mask);
% plotPolygons(ff_mine.xy_src_rect,'r.');

Z_gt = make_disparity(I1Rect,ff_sanity.xy_src,ff_sanity.xy_dst);
% figure(2);
subplot(n_rows,n_cols,4);imagesc2(Z_mine.*mask);title('Our Disparity');
subplot(n_rows,n_cols,5);imagesc2(Z_gt.*mask);title('True Disparity');

% plot the pixel error vs the height error
subplot(n_rows,n_cols,6);plot(vec_norms(dists),vec_norms(diff_3),'r.');
figure(2);
subplot(1,2,1);
hist(diff_3);legend({'x','y','z'});
subplot(1,2,2);
hist(my_dst-gt_dst);
legend({'x','y'});
%%
if showOneByOne
    %% visualize errors
    % idists = randperm(length(idists));
    for id = 1:length(idists)
        t = idists(id);
        clf;
        s1 = my_src(t,:);
        s2 = gt_dst(t,:);
        s1 = inflatebbox([s1 s1],45,'both',true);
        s2 = inflatebbox([s2 s2],45,'both',true);
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
end

end

