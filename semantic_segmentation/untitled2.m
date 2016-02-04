

images = {};
masks = {};
landmarks = {};
load ~/code/mircs/fra_db_2015_10_08.mat
addpath('~/code/mircs');
isTrain = [fra_db.isTrain];
for t = 1:length(fra_db)
    t
    %     if ~isTrain(t),continue,end
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    [I_sub,~,mouthBox,facePoly,I] = getSubImage2(conf,imgData,true);
    %[mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
    [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,true);
    faceMask = cropper(poly2mask2(facePoly,I),mouthBox);
    [groundTruth,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox);
        
    %clf; imagesc2(I); plotPolygons(I_sub
%     clf; displayRegions(I_sub,groundTruth);
%     continue;
    if isempty(groundTruth)
        groundTruth = false(size(faceMask));
    end
    images{t} = I_sub;
    curLandmarks(:,1:2) = bsxfun(@minus,curLandmarks(:,1:2),mouthBox(1:2));
    faceMask = faceMask &~groundTruth;
    curMask = double(groundTruth)+2*double(faceMask);
    masks{t} = curMask;
    landmarks{t} = curLandmarks;
end
% 