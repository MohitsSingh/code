function subs = getSubMouthImages(conf,imageSet)
sz = [50 50];
subs = zeros(sz,'uint8');
for k = 1:length(imageSet.labels)    
    imageInd = k 
    currentID = imageSet.imageIDs{imageInd};    
    I = getImage(conf,currentID);
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    lipRectShifted = imageSet.lipBoxes(imageInd,:);    
    box_c = round(boxCenters(lipRectShifted));    
    % get the radius using the face box.
    [r c] = BoxSize(faceBoxShifted);
    boxRad = (r+c)/2;    
    bbox = [box_c(1)-r/4,...
        box_c(2),...
        box_c(1)+r/4,...
        box_c(2)+boxRad/2];
    bbox = round(bbox);
        
    if (any(~inImageBounds(size(I),box2Pts(bbox))))        
        continue;
    end
    I_sub = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I_sub = imresize(I_sub,sz,'bilinear');
    I_sub = rgb2gray(I_sub);
    subs(:,:,k) = im2uint8(I_sub);
end
    %II = imresize(I_sub,2,'bicubic');
    