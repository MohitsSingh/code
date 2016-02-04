%% 
%%Robust face landmark estimation under occlusion. 
%%X.P. Burgos-Artizzu, P. Perona, P. Dollar. 
%%In ICCV'13, Sydney, Australia, December 2013
%%This example code shows how to load images and ground truth annotations used in paper

%Specify files containing training/testing images, load data
trFile='COFW_train.mat';testFile='COFW_test.mat';
load(trFile,'phisTr','IsTr','bboxesTr');
load(testFile,'phisT','IsT','bboxesT');

%% CONTENTS of files
%IsTr = training images = cell(1345,1) 
%IsT = testing images = cell(507,1) 
%phisTr = training ground truth = [1345 x 87]  where 1..29 is X position, 30..58 Y position, 59..87 occlusion bit
%phisT = testing ground truth = [507 x 87]  where 1..29 is X position, 30..58 Y position, 59..87 occlusion bit
%bboxesTr = training face bounding boxes = [1345 x 4]  (left, top, width, height)
%bboxesT = testing face bounding boxes = [507 x 4]  (left, top, width, height)
%
% NOTE 1: First 845 training images belong to LFPW dataset (unoccluded), only remaining 500 images are from COFW
% NOTE 2: Face bounding boxes are not perfect, as they simulate a real face detector
% See paper for more info/details
%% Example code to visualize annotations on a test image
%choose image to visualize

%show image

IsT = IsTr;
bboxesT = bboxesTr;
phisT = phisTr;
%%
imNum=13;
imagesc2(IsT{imNum}), hold on
%plot bounding box
plot([bboxesT(imNum,1) bboxesT(imNum,1)],[bboxesT(imNum,2) bboxesT(imNum,2)+bboxesT(imNum,4)],'-b','LineWidth',3)
plot([bboxesT(imNum,1) bboxesT(imNum,1)+bboxesT(imNum,3)],[bboxesT(imNum,2) bboxesT(imNum,2)],'-b','LineWidth',3)
plot([bboxesT(imNum,1)+bboxesT(imNum,3) bboxesT(imNum,1)+bboxesT(imNum,3)],[bboxesT(imNum,2) bboxesT(imNum,2)+bboxesT(imNum,4)],'-b','LineWidth',3)
plot([bboxesT(imNum,1) bboxesT(imNum,1)+bboxesT(imNum,3)],[bboxesT(imNum,2)+bboxesT(imNum,4) bboxesT(imNum,2)+bboxesT(imNum,4)],'-b','LineWidth',3)
%plot face landmarks (red occluded, green unoccluded)
occluded = find(phisT(imNum,59:end)==1);
unoccluded = find(phisT(imNum,59:end)==0);
plot(phisT(imNum,occluded),phisT(imNum,occluded+29),'.r','MarkerSize',15);
plot(phisT(imNum,unoccluded),phisT(imNum,unoccluded+29),'.g','MarkerSize',15);



