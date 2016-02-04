addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox');
% rmpath('/home/amirro/code/3rdparty/libsvm-3.17/matlab');
%  addpath('/home/amirro/code/3rdparty/SelectiveSearchCodeIJCV/');
addpath('/home/amirro/code/3rdparty/edgeBoxes/');
addpath('/home/amirro/code/3rdparty/uri/');
addpath('/home/amirro/code/3rdparty/ellipse/');
addpath('/home/amirro/code/3rdparty/uri/LogReg/');
% addpath('/home/amirro/code/3rdparty/linecurvature_version1b/');
addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
addpath('/home/amirro/code/3rdparty/LabelMeToolbox/');
addpath(genpath('/home/amirro/code/3rdparty/geom2d'));
addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));
addpath(genpath('/home/amirro/code/3rdparty/log4m'));
addpath(genpath('/home/amirro/code/3rdparty/markSchmidt/'));
%addpath(genpath('/home/amirro/code/3rdparty/object_bank/MATLAB_release/code/'));
% addpath('/home/amirro/code/3rdparty/face-release1.0-basic/'); % zhu + ramanan facial landmarks
addpath('/home/amirro/code/3rdparty/rendertext/');
% PASCAL VOC Images - used for negatives...
% addpath('/home/amirro/storage/data/VOCdevkit');
% addpath('/home/amirro/storage/data/VOCdevkit/VOCcode/');
addpath('/home/amirro/code/3rdparty/blocks/generics');
% VOCinit;
addpath(genpath('/home/amirro/code/3rdparty/hand_detector_code/code/skin_based_detector'));
addpath('/home/amirro/code/3rdparty');
% addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));
% addpath('/home/amirro/code/3rdparty/bresenham_mex/bresenham_mex');
addpath('/home/amirro/code/3rdparty/lineIntersection/');
addpath('/home/amirro/code/3rdparty/image_histogram_rgb_3d/');
addpath('/home/amirro/code/3rdparty/GCmex2.0');
addpath('/home/amirro/code/3rdparty/GCMex/');
addpath('/home/amirro/code/3rdparty/bspline/');
% addpath('/home/amirro/code/bow');
addpath(genpath('/home/amirro/code/3rdparty/poselets_matlab_april2013/'));
addpath('/home/amirro/code/3rdparty/sc');
handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
% rmpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
addpath(genpath('/home/amirro/code/3rdparty/toolbox-master/'));
% addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/');
% addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util');


% addpath(genpath('/home/amirro/code/3rdparty/exemplarsvm'));
% rmpath(genpath('/home/amirro/code/3rdparty/exemplarsvm'));


addpath(handDetectionBaseDir );
addpath(fullfile(handDetectionBaseDir,'code/globalPb'));


% DPM_path = '/home/amirro/code/3rdparty/voc-release4.01/';
% addpath(genpath(DPM_path));
vl_setup;

addpath(fullfile(pwd,'utils'));
addpath(fullfile(pwd,'features'));
addpath(fullfile(pwd,'face'));
% addpath(fullfile(pwd,'perf'));
addpath(fullfile(pwd,'vis'));
addpath(fullfile(pwd,'clustering'));
addpath(fullfile(pwd,'learning'));
addpath(fullfile(pwd,'geometric_relations'));
addpath(fullfile(pwd,'sets'));
addpath(fullfile(pwd,'demo'));
addpath(fullfile(pwd,'phases'));
addpath(genpath('/home/amirro/code/utils/'));

% vl_setup;
% addpath(genpath('D:\libsvm-3.12'));

opts.hands_locs_suff = 'hands_locs';
opts.hands_images_suff = 'hands_imgs';

% uncomment the following line if you wish to run the labeling tool.
%labeling_script;

% change this directory to where you put your standford40 dataset
conf.imgDir= '/home/amirro/storage/data/Stanford40/JPEGImages';
conf.imageSplitDir = '/home/amirro/storage/data/Stanford40/ImageSplits';
conf.xmlDir = '/home/amirro/storage/data/Stanford40/XMLAnnotations';
ext = '.jpg';

addpath('/home/amirro/code/common/');

actionsFileName = '/home/amirro/storage/data/Stanford40/ImageSplits/actions.txt';
[A,ii] = textread(actionsFileName,'%s %s');

f = fopen(actionsFileName);
A = A(2:end);

% for grab-cut:
% rmpath(genpath('/home/amirro/code/3rdparty/seg_transfer'));
