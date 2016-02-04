function [samples,labels] = sampleAroundMouth(imgData,regionSampler,useGT)
%mouthCorners = imgData.face_landmarks.xy(4:5,:);
mouthCenter = imgData.face_landmarks.xy(3,:);
mouth_rect_ratio = [.3 .2];
I = imgData.I;
% regionSampler.debugInfo = I;

if (~useGT)
    regionSampler.clearRoi();
else
    obj_poly = imgData.gt_obj;
    gt_region = poly2mask2(obj_poly,size2(I));
    z = fspecial('gauss',regionSampler.boxSize(1)/2, regionSampler.boxSize(1)/6);
    z = imfilter(double(gt_region),z);
    regionSampler.setRoiMask(z);
end
sampleBox = inflatebbox(mouthCenter,size2(I).*mouth_rect_ratio,'both',true);
samples = regionSampler.sampleOnGrid(sampleBox);
labels = samples(:,end);
samples = round(samples(:,1:4));