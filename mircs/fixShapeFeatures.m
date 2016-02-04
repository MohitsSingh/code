function [x1,x2,x3] = fixShapeFeatures(shapeFeats,n);
%x1 = cat(2,shapeFeats.shapeFeats1{im}(:);
x1 = cellfun(@(x) x(:), shapeFeats.shapeFeats1,'UniformOutput',false);
x1 = cat(2,x1{:});
x2 = cellfun(@(x) x(:), shapeFeats.shapeFeats2,'UniformOutput',false);
x2 = cat(2,x2{:});
ori = [shapeFeats.Orientation];
x3 = [shapeFeats.Area/n;shapeFeats.Eccentricity;shapeFeats.MajorAxisLength/sqrt(n);...
    shapeFeats.MinorAxisLength/sqrt(n);shapeFeats.Solidity;cosd(ori);sind(ori)];

if (nargout == 1)
    x1 = [x1;x2;x3];
end