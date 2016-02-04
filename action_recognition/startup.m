
% vl-feat
addpath('/home/amirro/code/3rdparty/vlfeat-0.9.18/toolbox');
addpath('/home/amirro/code/3rdparty/sc');
vl_setup;

% piotr dollar's functions
addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));

% my own functions and utilities
addpath(genpath('/home/amirro/code/utils/'));
addpath('/home/amirro/code/common');

% matconvnet + deep network model
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/examples');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/matlab');
vl_setupnn
conf.net = load('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/imagenet-vgg-m-2048.mat');

% selective search 
addpath(genpath('/home/amirro/code/3rdparty/SelectiveSearchCodeIJCV/'));

% deformable part models - include only what is needed
dpmPath = '/home/amirro/code/3rdparty/voc-release5';
addpath(dpmPath);
incl = {'context', 'bbox_pred', 'fv_cache', ...
    'bin', 'gdetect', 'utils', ...
    'car_grammar', 'person_grammar', ...
    'model', 'features', 'vis', ...
    'data', 'train', 'test', ...
    'external', 'star-cascade'};
for i = 1:length(incl)
    addpath(genpath(fullfile(dpmPath,incl{i})));
end
dpm_conf = voc_config();
% face detection module
load('~/code/3rdparty/dpm_baseline.mat');
conf.face_det_model = model;

%conf.baseDir = '~/storage/data/TUHOI';


