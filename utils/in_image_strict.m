function sel_ = in_image_strict(polys,sz)
sel_ = false(size(polys));
for k = 1:length(polys)
    xy = polys{k};
    sel_(k) = all(inImageBounds(sz,xy));
end