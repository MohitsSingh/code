load fra_db.mat

addpath('/home/amirro/code/3rdparty/sc');
addpath(genpath('/home/amirro/code/utils'));
figure(1);
imagesDir = '/home/amirro/storage/data/Stanford40/JPEGImages';
for t = 1:length(fra_db)
    clc
    fprintf('img %d out of %d\n',t,length(fra_db));
    imgData = fra_db(t);
    imagePath = fullfile(imagesDir,imgData.imageID);
    I = imread(imagePath);
    curPolygons = {imgData.objects.poly}; % may have more than one polygon per image,
    % currently just add them all up into one mask
    objects_mask = poly2mask2(curPolygons,size2(I));    
            
    clf; imagesc2(I); 
    plotPolygons(curPolygons,'r-','LineWidth',2);
    plotBoxes(imgData.faceBox);
    
    if boxRegionOverlap(imgData.faceBox,objects_mask)==0
        title('warning - no overlap between face and object!');
    end
    
    dpc(0);
    disp('hit any key for next image');
    
end
    
    %displayRegions(I,objects_mask);
    


