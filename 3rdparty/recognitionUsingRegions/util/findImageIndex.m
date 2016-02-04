function ind = findImageIndex(imageData,imageID)    
    ind = find(cellfun(@any,strfind({imageData.imageID},imageID)),1,'first');
end