function [shapeFeats] = getShapeFeatures(conf,I,regions)

% get bounding boxes for all regions
I =  im2single(I);
for iRegion = 1:length(regions)
    %     iRegion
    [subI,x] = getShapeAndSubImage(I,regions{iRegion});
    subI = imresize(subI,[80 80],'bilinear');
    curHOG = vl_hog(subI,8);
    curMask = imresize(single(x),[80 80],'bilinear');
    curHOG_mask = vl_hog(curMask,8);
    shapeFeats1{iRegion} = curHOG; % regular hog
    shapeFeats2{iRegion} = curHOG_mask; % hog of mask
    masks{iRegion} = imresize(single(x),[10 10],'bilinear'); % mask
end
shapeFeats.shapeFeats1 = shapeFeats1;
shapeFeats.shapeFeats2 = shapeFeats2;
shapeFeats.masks = masks;
rprops = {};
for k = 1:length(regions)
    rprops{k} = regionprops(regions{k},'Area','Eccentricity','Solidity','MajorAxisLength','MinorAxisLength',...
        'Orientation');
end

rprops = cat(1,rprops{:});
shapeFeats.Area = cat(1,[rprops.Area]);
shapeFeats.Eccentricity = cat(1,[rprops.Eccentricity]);
shapeFeats.Solidity = cat(1,[rprops.Solidity]);
shapeFeats.MajorAxisLength = cat(1,[rprops.MajorAxisLength]);
shapeFeats.MinorAxisLength = cat(1,[rprops.MinorAxisLength]);
shapeFeats.Orientation = cat(1,[rprops.Orientation]);

end
