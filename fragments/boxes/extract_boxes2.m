function extract_boxes2(imageID)
%EXTRACT_BOXES preprocess each of the images in the given list to extract boxes
%and tiny images
%   Detailed explanation goes here

addpath('/home/amirro/data/VOCdevkit/VOCcode/');
VOCinit;
outDir = '~/boxes/';
if (~exist(outDir,'dir'))
    mkdir(outDir);
end

boxesFileName = fullfile(outDir,imageID);
imPath = sprintf(VOCopts.imgpath,imageID);
I = imread(imPath);
boxes = SelectiveSearchBoxes(I); %#ok<NASGU>
save(boxesFileName,'boxes');

% end