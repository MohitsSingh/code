function label = ucm22label(ucm2, th)
% function label = ucm22label(ucm2, th)
%
% This function inputs ucm2 maps and outputs an indexing of the associated
% regions extracted by ucm2 with threshold th.
%
% Copyright @ Chunhui Gu, April 2009

labels2 = bwlabel(ucm2 < th);
label = labels2(2:2:end, 2:2:end);