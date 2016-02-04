function k = findImageIndex(newImageData,imageID)
    if isstruct(imageID)
        imageID = imageID.imageID;
    end
    k = find(cellfun(@any,strfind({newImageData.imageID},imageID)));
end