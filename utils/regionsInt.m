function [ints,a1,a2] = regionsInt(regions1,regions2)
%REGIONSINT Return intersection area of all region pairs.
regions1 = cellfun2(@col,regions1);
regions1 = cat(2,regions1{:});
regions2 = cellfun2(@col,regions2);
regions2 = cat(2,regions2{:});
ints = single(regions1)'*single(regions2);
a1 = sum(regions1,1);
a2 = sum(regions2,1);
end

