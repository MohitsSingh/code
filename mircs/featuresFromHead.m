function [ feats ] = featuresFromHead(I, poseData, net , layers,net_deep,layers_deep,partOfHead)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
if (nargin < 4)
    layers = 16;
end
faceBox = poseData.faceBox;
faceScale = faceBox(4)-faceBox(2);
types = {'eye_left','eye_right','mouth','ear_left','ear_right'};
indices = [6 19 9 5 18];
valids = poseData.landmarks(indices,end);
boxSize = faceScale*partOfHead;
boxes = inflatebbox(poseData.landmarks(indices,[1 2 1 2]),boxSize,'both',true);
crops = multiCrop2(I,round(boxes));
%indices = mat2cell2(indices,1,length(indices));
valids= mat2cell2(valids,[1,length(valids)]);
subImages = struct('type',types,'img',crops,'valid',valids);
feats = struct('type',{},'feat',{});
for t = 1:length(subImages)
    feats(t).type = subImages(t).type;
    sub_img = subImages(t).img;
    feat = extractDNNFeats(sub_img,net,layers);
    feats(t).feat = feat.x;
end
t_ = length(feats);
for t = 1:length(subImages)
    k = t_+t;
    feats(k).type = [subImages(t).type '_deep'];
    sub_img = subImages(t).img;
    clf; imagesc2(sub_img);pause;
    feat = extractDNNFeats(sub_img,net_deep,layers_deep);
    feats(k).feat = feat.x;
end

% add head and extended head region...
I_face = cropper(I,round(faceBox));
I_face_extended = cropper(I,round(inflatebbox(faceBox,2,'both',false)));
feats_face = extractDNNFeats(I_face,net,layers); % fc6, fc7
feats_face_ext = extractDNNFeats(I_face_extended,net,layers); % fc6, fc7
feats_face_deep = extractDNNFeats(I_face,net_deep,layers_deep); % fc6, fc7
feats_face_ext_deep = extractDNNFeats(I_face_extended,net_deep,layers_deep); % fc6, fc7
t = length(feats)+1;
feats(t).feat = feats_face.x;
feats(t).type = 'face';
t = t+1;
feats(t).feat = feats_face_ext.x;
feats(t).type = 'face_ext';
t = t+1;
feats(t).feat = feats_face_deep.x;
feats(t).type = 'face_deep';
t = t+1;
feats(t).feat = feats_face_ext_deep.x;
feats(t).type = 'face_ext_deep';
%     function poseData = augmentPoseData(I,poseData,faceScale)
% get a bounding box around the eye and the mouth, etc, crop out
% the image.
%         error('not implemented yet!');
%     end
end