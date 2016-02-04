function facialLandmarkData=initFacialLandmarkData()
addpath('/home/amirro/code/3rdparty/rcpr_v1');
load regModel_frontal
load regModel_profile
T=100;K=15;L=20;RT1=5;
ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);
prm=struct('thrr',[-1 1]/5,'reg',.01);
occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',3,'th',.5);
regPrm_frontal = struct('type',1,'K',K,'occlPrm',occlPrm,...
    'loss','L2','R',0,'M',5,'model',regModel_frontal.model,'prm',prm);
regPrm_profile = struct('type',1,'K',K,'occlPrm',occlPrm,...
    'loss','L2','R',0,'M',5,'model',regModel_profile.model,'prm',prm);
facialLandmarkData.prunePrm=struct('prune',1,'maxIter',2,'th',0.15,'tIni',10);

facialLandmarkData.regModel_frontal = regModel_frontal;
facialLandmarkData.regPrm_frontal = regPrm_frontal;
facialLandmarkData.regModel_profile = regModel_profile;
facialLandmarkData.regPrm_profile = regPrm_profile;
facialLandmarkData.T=100;
facialLandmarkData.K=15;
facialLandmarkData.L=20;
facialLandmarkData.RT1=30;
facialLandmarkData.ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);

load ~/storage/misc/landmark_training_data %faces_and_landmarks_frontal faces_and_landmarks_profile
%%

load ~/storage/misc/landmark_training_data_all_c;

% zero_borders = true;
% wSize = 48;
% Is_frontal = {faces_and_landmarks_frontal.I};
% nFrontal = length(Is_frontal);
% XX_frontal = getImageStackHOG(Is_frontal,wSize,true,zero_borders );
% Is_profile = {faces_and_landmarks_profile.I};
% XX_profile = getImageStackHOG(Is_profile,wSize,true,zero_borders );
% II = [Is_frontal,Is_profile];
% XX = [XX_frontal XX_profile];
% kdtree = vl_kdtreebuild(XX,'Distance','L1');
% face_comps = ones(size(II));
% face_comps(length(Is_frontal)+1:end) = 2;
% all_c = zeros(size(face_comps));
facialLandmarkData.all_c = [[faces_and_landmarks_frontal.c],[faces_and_landmarks_profile.c]];
facialLandmarkData.nFrontal = nFrontal;
