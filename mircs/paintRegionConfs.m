function Z = paintRegionConfs(I,regions,regionConfs,T)
if (nargin < 4)
    T = 5000;
end
Z = -inf(dsize(I,1:2));
for k = 1:length(regions)
    if (nnz(regions{k}) < T)
        Z(regions{k}) = regionConfs(k);
    end
end
end