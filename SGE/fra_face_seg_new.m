function res = fra_face_seg_new(conf,I,reqInfo,pipelineStruct)


if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    res.conf = conf;
    addpath('~/code/SGE');
    addpath(genpath('~/code/utils'));    
    addpath('/home/amirro/code/mircs/');
%     res.fra_db = s40_fra;
    return;
end

    
res = [];
[I_orig,I_rect] = getImage(conf,I);
fra_struct = face_detection_to_fra_struct(conf,pipelineStruct.funs(1).outDir,I);

if (all(isinf(fra_struct.faceBox)))
    res = [];
    return;
end
fra_struct.faceBox = fra_struct.faceBox+I_rect([1 2 1 2]);
conf.get_full_image = true;

if (~fra_struct.valid)
    return;
end
roiParams.centerOnMouth = false;
roiParams.infScale = 1.5;
roiParams.absScale = 192;
[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_struct,roiParams);
origDir = pwd;
cd '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';install;
[candidates, ucm2,valid] = im2mcg(I,'accurate',false);
res.valid = valid;
res.candidates = candidates;
res.ucm2 = ucm2;
res.roiBox = roiBox;
cd(origDir);