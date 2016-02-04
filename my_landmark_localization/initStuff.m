if (~exist('initialized','var'))
    addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
    addpath(genpath('~/code/utils/'));
    addpath(genpath('/home/amirro/code/3rdparty/toolbox-master/'));
    addpath ~/code/mircs/features
    addpath ~/code/common/
    addpath(genpath('/home/amirro/code/3rdparty/svm-struct/'));    
%     addpath('X:\code\3rdparty\exemplarsvm\features');
    addpath('~/code/3rdparty/sc');
    addpath(genpath('~/code/3rdparty/UGM'));

    addpath /home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox/
    vl_setup
    trFile='COFW_train_color.mat';testFile='COFW_test_color.mat';
    load(trFile,'phisTr','IsTr','bboxesTr');    
    [phisTr,IsTr,bboxesTr] = prepareData(phisTr,IsTr,bboxesTr);
    load(testFile,'phisT','IsT','bboxesT');
    [phisT,IsT,bboxesT] = prepareData(phisT,IsT,bboxesT);
    initialized = true;
end