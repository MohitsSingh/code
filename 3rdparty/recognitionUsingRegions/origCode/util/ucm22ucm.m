function ucm = ucm22ucm(ucm2, th, out)
% function ucm = ucm22ucm(ucm2, th, out)
%
% This function converts ucm2 signal into ucm.
%
% Copyright @ Chunhui Gu, April 2009

ucm = double(hthin(ucm2(3:2:end, 3:2:end), 8, out));

if (nargin > 1)
    ucm = ucm .* (ucm >= th);
end