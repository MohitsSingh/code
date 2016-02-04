function m = multiResize(m,sz)
if (isscalar(sz))
    sz = [sz sz];
end
m = cellfun2(@(x) imResample(x, sz, 'bilinear'), m);
end