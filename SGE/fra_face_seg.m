function res = fra_face_seg(conf,I,reqInfo,override)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    res.conf = conf;
    addpath('~/code/SGE');
    addpath(genpath('~/code/utils'));
    
    %     install;
% % %     load ~/code/mircs/fra_db.mat;
% % %     addpath('/home/amirro/code/mircs/');
% % %     res.fra_db = fra_db;


%     install;
    load ~/code/mircs/s40_fra.mat;
    addpath('/home/amirro/code/mircs/');
    res.fra_db = s40_fra;
    return;
end
   
res = [];
fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
if (~curImageData.valid)
    return;
end
roiParams.infScale = 1.5;
roiParams.absScale = round(192*1.5/1.5);
roiParams.centerOnMouth = false;
curImageData = switchToGroundTruth(curImageData);
[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
origDir = pwd;
cd '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';install;
[candidates, ucm2] = im2mcg(I,'accurate',false);
res.candidates = candidates;
res.ucm2 = ucm2;
res.roiBox = roiBox;
cd(origDir);