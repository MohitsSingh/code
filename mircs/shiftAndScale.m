function x = shiftAndScale(x,offsets,factors)
% 
if (iscell(x))
    for t = 1:length(x)
        x{t} = bsxfun(@minus,x{t},offsets(t,1:2))*factors(t);
    end
else
    x =  bsxfun(@times,bsxfun(@minus,x,offsets(:,1:2)),factors)
end
end