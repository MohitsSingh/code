
if ~exist('A')
    path_to_lwc = 'LWC_1520.mat';
    A = load(path_to_lwc);
    addpath(genpath('~/code/utils'));
    addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
    addpath(genpath('~/code/3rdparty/ImGRAFT-master/'));
    addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox');vl_setup;
    addpath('Validation');
    addpath(genpath('/home/amirro/code/3rdparty/dsp-code'));
%     addpath('~/code/3rdparty/sls_distribution.1.1/');
end

%addpath('/home/amirro/code/3rdparty/gridfitdir/');
% generate ground-truth cameras
gtFileName = '~/dad_with_gt.mat';
if exist(gtFileName,'file')
    load(gtFileName);
else
    world_to_cam_samples = generateGroundTruth(A);
    save(gtFileName,'world_to_cam_samples');
end
% convert cameras to matlab-friendly format
cameras = getCameraParams(world_to_cam_samples);

% apply reconstruction
do_reconstruction;