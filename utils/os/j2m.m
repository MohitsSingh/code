function [ f ] = j2m(baseDir, id ,suffix)
%J2M replace ".jpg" suffix with ".mat" suffix, and append to
% directory b.
%   Detailed explanation goes here
if (isstruct(id))
    id = id.imageID;
    [~,name,ext] = fileparts(id);
    id = [name ext];
end

if (nargin < 3)
    suffix = '.mat';
end
[pathstr,name,ext] = fileparts(id);
if (isempty(ext))
    id = [name '.jpg'];
end
f = fullfile(baseDir,[name,suffix]);

end

