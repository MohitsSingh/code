function [ gtboxes ] = getgtboxes(globalopts,currentid,cls,ignorehard,ignoretrunc)
%getgtboxes summary of this function goes here
%   detailed explanation goes here
if (nargin < 5)
    ignoretrunc = false;
end
if (nargin < 4)
    ignorehard = false;
end
if (nargin < 3)
    cls = [];
end

rec = pasreadrecord(getrecfile(globalopts,currentid));
if (~isempty(cls))
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
else
    clsinds=1:length(rec.objects);
end

isdifficult = [rec.objects(clsinds).difficult];
if (ignorehard)
    isdifficult = false(size(isdifficult));
end
istruncated = [rec.objects(clsinds).truncated];%this effectively
if (ignoretrunc)
    istruncated = false(size(istruncated));
end

% toggles usage of truncated examples. 'false' means they are used.
% get bounding boxes
gtboxes = cat(1,rec.objects(clsinds(~isdifficult & ~istruncated)).bbox);

end

