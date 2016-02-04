function f = normalizeData2(f)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    f = bsxfun(@minus,f,min(f,[],2));
    f = bsxfun(@rdivide,f,max(f,[],2));
    f = 2*(f-.5);
end

