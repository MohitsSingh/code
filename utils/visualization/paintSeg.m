function [segImage,c] = paintSeg(I,labelImage)
%PAINTSEG Summary of this function goes here
%   Detailed explanation goes here
%     segImage = reshape(I,[],3);
%     sz = size(I);
%     labelImage = labelImage(:);
%     u = unique(labelImage);
%     I = im2uint8(I);
    c = {};
    for k = 1:size(I,3)
        c{k} = regionprops(labelImage,I(:,:,k),'MeanIntensity','PixelIdxList');
    end
    
    segImage = zeros(size(I),class(I));
    for k = 1:size(I,3)
        z = zeros(size(segImage,1),size(segImage,2),class(I));
        cc = c{k};
        for q = 1:length(cc)
            z(cc(q).PixelIdxList) = cc(q).MeanIntensity;
        end
        segImage(:,:,k) = z;
    end
end

