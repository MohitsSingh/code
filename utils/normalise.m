function [I,I_min,I_max] = normalise(I,I_min,I_max)
% if (nargin < 2)
if numel(I)==1
    I_min = I;I_max = I;
    return;
end
% end
I_min = min(I(~isinf(I)));
I = I-I_min;
I_max = max(I(~isinf(I)));
I = I/I_max;
I(isinf(I)) = 0;
end