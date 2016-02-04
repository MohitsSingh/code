function [I] = addBorder(I,borderWidth,borderColor)
%ADDBORDER Summary of this function goes here
%   Detailed explanation goes here
    m = false(size(I,1),size(I,2));
    m(1:borderWidth,:) = 1;
    m = max(m,flipud(m));
    m(:,1:borderWidth) = 1;
    m = max(m,fliplr(m));
    for iChannel = 1:size(I,3)
        c = I(:,:,iChannel);
        %c(m) = .3*c(m)+.7*borderColor(iChannel);
        c(m) = borderColor(iChannel);
        I(:,:,iChannel) = c;
    end
end