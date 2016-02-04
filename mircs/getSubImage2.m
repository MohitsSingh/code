function [I_sub,faceBox,mouthBox,face_poly,I] = getSubImage2(conf,imgData,use_gt,other_bb,inflateFactor)
I = getImage(conf,imgData);
if nargin < 5
    inflateFactor=[];
end
   
if nargin < 3
    use_gt = false;
end
if use_gt
    faceBox = imgData.faceBox;
    mouthCenter = imgData.landmarks_gt.xy(3,:);
else
    if nargin < 4 || isempty(other_bb)
        faceBox = imgData.faceBox_raw;
    else
        faceBox = other_bb;
    end
    mouthCenter = imgData.landmarks.xy(3,:);
end

objects = imgData.objects;
iFace = strcmp({objects.name},'face');% imgData
face_poly = objects(iFace).poly;
h = faceBox(3)-faceBox(1);
if isempty(inflateFactor)
    mouthBox = round(inflatebbox(mouthCenter,h,'both',true));
else
    mouthBox = inflatebbox(mouthCenter,inflateFactor*h,'both',true);
%     mouthBox = round(inflatebbox(mouthBox,inflateFactor,'both',false));
end
I_sub = cropper(I,mouthBox);