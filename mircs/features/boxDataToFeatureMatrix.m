function [faceFeats,mouthFeats,imgFeats,imgLabels,boxFeats,boxLabels,img_inds,ovps] = boxDataToFeatureMatrix(imgDatas,boxesData,imageData,T_ovp)

boxesData = cat(2,boxesData{:});

imageData = cat(2,imageData{:});
imgLabels = [imgDatas.classID];
faceFeats = [imageData.face];
mouthFeats = [imageData.mouth];
imgFeats = [imageData.global];

img_inds = [boxesData.imgIndex];
boxLabels = imgLabels(img_inds);
ovps = [boxesData.ovp];
boxLabels(ovps<T_ovp) = 0;

boxFeats = cat(2,boxesData.feats);
% 
% boxFeats = boxFeats(:,1:5:end);
% boxLabels = boxLabels(:,1:5:end);