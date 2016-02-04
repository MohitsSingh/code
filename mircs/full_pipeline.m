function [ imageData ] = full_pipeline( conf, imageData )
%FULL_PIPELINE Summary of this function goes here
%   Detailed explanation goes here

%TODO : add caching by image data or image id
% imPath = '/home/amirro/storage/data/drinking_extended/straw/31123_405053918336_1072549_n.jpg';
imageData = imPath;
% I = imread(imPath);
[I,I_rect] = getImage(conf,imageData);

seg_data = gpb_segmentation(conf,imageData);
imageData.imageID = imageData;
imageData.segData = seg_data;
% detect faces
[~,face_boxes] = face_detection(I);
face_boxes = inflatebbox(face_boxes,[1.5 1.5],'both',false);
face_boxes = clip_to_image(face_boxes,I);
face_ims = multiCrop(conf,I,face_boxes);
% extract landmarks for each detected face
n = 0;
for k = 1:length(face_ims)
    curIm = face_ims{k};
    curBox = face_boxes(k,:);
    landmarks = extractLandmarks(conf,curIm);
    landmarks = landmarks([landmarks.isvalid]);
    [faceLandmarks] = landmarks2struct_3(landmarks,curBox(1:2));
    for t = 1:length(faceLandmarks)
        n = n+1;
        imageData.faceLandmarks(n) = faceLandmarks(t);
    end
end

for k = 1:length(faces)
    [occlusionPatterns,regions,face_mask,mouth_mask] = getOcclusionPattern_2(conf,imageData,faceLandmarks(k));
    
    imageData.faceScore=0;
    [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,imageData,1.5,1);
    curLandmarks = faceLandmarks(k);
    occlusionPattern.mouth_poly = mouth_poly;
    occlusionPattern.face_poly = face_poly;
    occlusionPattern.occlusionPatterns = occlusionPatterns;
    occlusionPattern.regions = regions;
    occlusionPattern.face_mask = face_mask;
    occlusionPattern.mouth_mask = mouth_mask;
    occlusionPattern.faceBox = curLandmarks.faceBox;
    clear occlusionData;
    [occlusionData.occludingRegions,occlusionData.occlusionPatterns,occlusionData.rprops] = getOccludingCandidates_2(I,occlusionPattern);
    imageData.occlusionData(k) = occlusionData;
end

% 2.if nothing found, run face detector. otherwise, run face detectors within
% upper body region.
% 3. run zhu/ramanan on each region.

% run segmentation
% infer occlusions
% apply detectors
% make final decision

%         figure,imshow(curIm); hold on; plotPolygons(polys)

imwrite(I,'~/face_1.jpg');