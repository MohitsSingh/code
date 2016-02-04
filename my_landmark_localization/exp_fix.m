function b = exp_fix(b)
    d = b-min(b(:));
    b = exp(d);
%     b = exp(b);
end