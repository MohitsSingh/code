function newImageData = augmentGT(newImageData,m)
    nonEmptyCount = 0;
    for k = 1:length(m)
        imageInd = findImageIndex(newImageData,m(k).imageID);
        if (isempty(imageInd))
            continue;
        end
        nonEmptyCount = nonEmptyCount+1
        newImageData(imageInd).extra = m(k);        
    end
end