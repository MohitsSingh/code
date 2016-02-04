function montage3(A)
%MONTAGE3 Summary of this function goes here
%   Detailed explanation goes here
hasChn = true;
if (iscell(A))
    sizes = cellfun2(@size2,A);
    sizes = mean(cat(1,sizes{:}),1);
    A = cellfun2(@(x) imresize(x,sizes([1 1]),'bilinear'),A);
    
    if (length(size(A{1})) == 3)
        A = cat(4,A{:});
        hasChn = true;
    else
        A = cat(3,A{:});
    end
end

montage2(A,struct('hasChn',hasChn));
end

