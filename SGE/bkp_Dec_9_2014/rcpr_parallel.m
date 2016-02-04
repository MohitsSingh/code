function res = rcpr_parallel(conf,I,reqInfo)

if (nargin == 0)
    addpath('~/code/SGE');
    cd ~/code/mircs;
    initpath;
    config;
    addpath(genpath('~/code/utils'));
    addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
    addpath('/home/amirro/code/3rdparty/rcpr_v1');
    cd '/home/amirro/code/3rdparty/rcpr_v1';
    load('data/COFW_train','bboxesTr');
    load('data/rcpr.mat','regModel','regPrm','prunePrm');
    load ~/storage/misc/imageData_new_sub;
    cd ~/code/mircs;
    res = struct('bboxesTr',bboxesTr,'regModel',regModel,'regPrm',regPrm,'prunePrm',prunePrm,...
        'newImageData',newImageData,'conf',conf);
    return;
end
I = imread(I);
ds = imgdetect(I,model,-.5);
res = ds(nms(ds,.5),:);
