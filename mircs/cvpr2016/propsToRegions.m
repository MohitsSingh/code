function regions = propsToRegions(props,sz)
regions = {};
for t = 1:length(props)
    r = false(sz);
    r(props(t).PixelIdxList) = 1;
    regions{t} = r;
end