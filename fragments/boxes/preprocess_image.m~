function preprocess_image(globalOpts,imageID)
%PREPROCESS_IMAGE Extracts subwindows and descriptors
% for those windows from the image.

% tinyImagesFileName = sprintf(VOCopts.exfdpath,[imageID '_tinyimages']);
% create tiny image from each box...
% if (exist(tinyImagesFileName,'file'))
%     load(tinyImagesFileName);
% else
% extract boxes using selective search
boxesFileName = getBoxesFile(globalOpts,imageID);
if (exist(boxesFileName,'file'))
    load(boxesFileName);
else
    
    %imPath = sprintf(globalOpts.VOCopts.imgpath,imageID);
    imPath = getImageFile(globalOpts,imageID);
    I = imread(imPath);
    boxes = SelectiveSearchBoxes(I); %#ok<NASGU>
    save(boxesFileName,'boxes');
end

