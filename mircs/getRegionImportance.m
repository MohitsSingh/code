function [I,U,detections]=getRegionImportance(conf, curID,w ,featureExtractor);
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
I = getImage(conf,curID);

f = featureExtractor.extractFeatures(I);
I = getImage(conf,curID);
I = makeImageSquare(I);
w = w(1:end-1)';
[U] = doMaskSearch(im2double(I),featureExtractor,w);
detections = 0;
U = [im2double(I) (sc(cat(3,U,im2double(I)),'prob_jet'))];

% U = imResample(U,256/size(U,1),'bilinear');






