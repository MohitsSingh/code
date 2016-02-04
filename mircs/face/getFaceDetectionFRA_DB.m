function [I,face_boxes] = getFaceDetectionFRA_DB(conf,curImgData)


R = j2m('~/storage/fra_faces_baw',curImgData);
load(R);
res.detections = res.detections(3);
face_boxes = cat(1,res.detections.boxes);
roiParams.centerOnMouth = false;
roiParams.infScale = 1.5;
roiParams.absScale = 192;
conf.get_full_image = true;
[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImgData,roiParams); %

