function [ I ] = seg2col( I,R )
%SEG2COL Summary of this function goes here
%   Detailed explanation goes here
rprops = regionprops(R,'PixelIdxList');

for cc = 1:size(I,3)
    currentChannel = I(:,:,cc);
    for k = 1:length(rprops)
        m = mean(currentChannel(rprops(k).PixelIdxList));
        currentChannel(rprops(k).PixelIdxList) = m;
    end
    I(:,:,cc) = currentChannel;
end


end

