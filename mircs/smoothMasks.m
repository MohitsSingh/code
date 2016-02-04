function newMasks = smoothMasks(masks)
newMasks = {};
for k = 1:length(masks)
    k
    curMask = masks{k};
    r = regionprops(curMask,'Area','PixelIdxList');
    if (~isempty(r)==0)
        warning(['no segments found for image ' num2str(kk)]);
        continue;
    end
    % get the largest segment...
    [~,imax] = max([r.Area]);
    Z = zeros(size(curMask));
    Z(r(imax).PixelIdxList) = 1;
    %     Z = imfilter(im2double(Z),fspecial('gauss',9,3));
    %     minVal = .3;
    %     maxVal = .99;
    Z = imfilter(im2double(Z),fspecial('gauss',99,3));
    Z = log(Z);
    minVal = -15;
    maxVal = log(.8);
    newMasks{k} = (Z<maxVal & Z>minVal);
    %     imagesc(newMasks{k});
    %     pause(.1);
end