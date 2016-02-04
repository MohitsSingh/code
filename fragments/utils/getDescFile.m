function [ descPath ] = getDescFile(globalOpts,currentID,s)
%GETDESCFILE Summary of this function goes here
%   Detailed explanation goes here
% s specifies if these are sparse descs, sampled for vocabulary,
% or dense...
descPath = sprintf(globalOpts.featPath,currentID);
if (nargin > 2 && s)
    descPath = strrep(descPath,'.mat','_sample.mat');
end
end

