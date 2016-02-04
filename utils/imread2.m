function I = imread2(p)
[I,cmap] = imread(p);
if (~isempty(cmap))
    I = ind2rgb(I,cmap);
end
if (length(size(I))< 3)
    I = cat(3,I,I,I);
end
end