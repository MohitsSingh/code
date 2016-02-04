function I = clamper(I,rect)
if (numel(rect) == 4)
    rect =  poly2mask2(box2Pts(rect),size2(I));
end
for iC = 1:size(I,3)
    I_ = I(:,:,iC);
    I_(~rect) = 0;
    I(:,:,iC) = I_;
end
end