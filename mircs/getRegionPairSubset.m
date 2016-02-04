function [regions,pairs] = getRegionPairSubset(conf,currentID,maxRegions)
[regions,regionOvp,G] = getRegions(conf,currentID);
[ii,jj] = find(G);
pairs = [ii';jj'];
pairs = vl_colsubset(pairs,maxRegions);
pairs = pairs';
end