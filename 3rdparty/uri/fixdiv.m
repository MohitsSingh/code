function out = fixdiv(in, val)

if (nargin < 2)
    val = 1;
end

out = in;
out(in == 0) = val;
