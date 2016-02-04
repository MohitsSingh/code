

if ~exist('A')
    path_to_lwc = 'LWC_1520.mat';
    A = load(path_to_lwc);
    LWC = A.LWC;
    addpath(genpath('~/code/utils'));
    addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
    addpath(genpath('~/code/3rdparty/ImGRAFT-master/'));
    addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox');vl_setup;
    addpath('Validation');
    addpath(genpath('/home/amirro/code/3rdparty/dsp-code'));
    addpath(genpath('/home/amirro/code/3rdparty/imrender/'));        
    global matchedDir;
    global origDir;
    matchedDir = '/home/amirro/code/clouds/matched_pairs';
    origDir = '/home/amirro/code/clouds/Images_divided_by_maxValue';        
    addpath('~/code/3rdparty/deepmatching_1.0.2_c++/');   
%     addpath('~/code/3rdparty/sls_distribution.1.1/');
end


%addpath('/home/amirro/code/3rdparty/gridfitdir/');
% generate ground-truth cameras
% gtFileName = '~/camera_data.mat';
% save(gtFileName,'world_to_cam_samples','cameras')
% if exist(gtFileName,'file')
%     load(gtFileName);
% else
%%
z_offset = 1;
my_offset = 0;
[world_to_cam_samples,patterns] = generateGroundTruth(A,my_offset,z_offset);
cameras = struct('camMatrix',cellfun2(@(x) x',{world_to_cam_samples.projMat}));
B = load('ConvectiveCloud_Images');
B.Img = cellfun2(@(x) imrotate(x,-90),B.Img);
%%
%%
global catchSystematicError
catchSystematicError = false;

% apply reconstruction
do_reconstruction;

%%
% for t = 1:17
%     clf; imagesc2(R.Img{1,t})
%     dpc
% end

%% 

load('Aviad_Calib/ground_points_and_values.mat'); % Points
load('Aviad_Calib/Calib_images.mat'); % Img
% rotate all images by -90 degrees

Img = cellfun2(@(x) imrotate(x,-90),Img);
% select a single point manually in images 1,5,9
p1 = [261 149];
p5 = [261 91];
p9 = [73 75];
% use cameras 1 and 5 to reconstruct the point
w15 = triangulate(p1,...
        p5,...
        cameras(1).camMatrix, cameras(5).camMatrix);
% use cameras 5 and 9 to reconstruct the point    
w59 = triangulate(p5,...
        p9,...
        cameras(5).camMatrix, cameras(9).camMatrix);
% compare the two reconstructed points    
w59-w15
vec_norms(w59-w15)

% show some statistics on the norm of the difference between reconstruction
% points vs the error in image to image matching. 
% do this for the mean point of the selectec calibration points. 
% this can be computed analytically but we're doing it empirically here.

%%
p = mean(Points(:,1:3));
cam1 = cameras(1);
cam2 = cameras(5);
p1 = project_to_image(p,cam1);
p2 = project_to_image(p,cam2);
error_range = -5:.2:5;
[x,y] = meshgrid(error_range,error_range);
shp = size(x);
p2_with_error = bsxfun(@plus,p2,[x(:) y(:)]);
image_diffs = vec_norms([x(:) y(:)]);
% "real" triangulation
P_real = triangulate(p1,p2,cam1.camMatrix, cam2.camMatrix);
vec_norms(P_real-p)
P_with_error = triangulate(repmat(p1,length(x(:)),1),p2_with_error,cam1.camMatrix, cam2.camMatrix);
world_diffs = vec_norms(bsxfun(@minus,P_real,P_with_error));
world_diffs = reshape(world_diffs,shp);
imagesc(world_diffs);
% ticks = cellfun2(@(x) num2str(x),mat2cell2(error_range,[1 length(error_range)]));
% set(gca,'XTick',1:length(error_range),'XTickLabel',ticks);
% set(gca,'YTick',1:length(error_range),'YTickLabel',ticks);
contour(x,y,world_diffs)
ylabel('y matching error'); xlabel('x matching error'); zlabel('recon. error norm')
colorbar
title('reconstruction error norm vs matching displacement(offset=4)');


plot(image_diffs(:),world_diffs(:),'r.'); grid on; xlabel('image matching error norm');
ylabel('reconstruction error norm');
% plot(image_diffs,world_diffs,'r+')




%%
n ={};
for t = 1:length(camera_data)
    f = camera_data(t);
    n{t} = struct(f);
end

n = [n{:}];

camera_data = n;
