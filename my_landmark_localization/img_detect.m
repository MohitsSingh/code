function [res] = img_detect(param,curImg)
if (length(size(curImg))==2)
    curImg = cat(3,curImg,curImg,curImg);
end

[res] = img_detect2(param,param.patchDetectors,curImg);

