function [ ids] = getNonPersonIds(VOCopts,invert,subset)
%GETNONPERSONIDS Summary of this function goes here
%   Detailed explanation goes here
%%
if (nargin < 2)
    invert = false;
end
if (nargin < 3)
    subset = '_train';
end
cls = 15; % class of person;
[ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} subset]),'%s %d');

if (invert)
    ids = ids(t==1); % persons yes gratis
else
    ids = ids(t==-1); % persons non gratis
end
end