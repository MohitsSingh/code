function goods = inMask( xy, mask)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if (size(xy,2)==2)
    xy = xy';
end
f1_inds = sub2ind2(size2(mask),round(xy([2 1],:)'));
goods = mask(f1_inds);

end

