function [regions,selection] = ezRemove(regions,I_sub,minAbsSize,maxRelSize)
A = cellfun(@nnz,regions);
selection = A >= minAbsSize & A <= prod(size2(I_sub))*maxRelSize;
regions = regions(selection);
%regions(cellfun(@nnz,regions) < minAbsSize) = [];
%regions(cellfun(@nnz,regions) > prod(size2(I_sub))*maxRelSize) = [];
