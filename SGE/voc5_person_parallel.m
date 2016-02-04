function res = voc5_person_parallel(conf,I,reqInfo)

if (nargin == 0)            
    addpath('~/code/SGE');
    addpath(genpath('~/code/utils'));
    cd '/home/amirro/code/3rdparty/voc-release5';
    startup;    
    load('/home/amirro/code/3rdparty/voc-release5/VOC2007/person_grammar_final.mat');
    res = struct('model',model);
    return;
end

I = imread(I);
ds = imgdetect(I,reqInfo.model,-.5);
res = ds(nms(ds,.3),:);
