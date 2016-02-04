function [ m ] = clusters2Images( c ,borderColor)
%CLUSTERS2IMAGES Summary of this function goes here
%   Detailed explanation goes here
m = {};
max_sz = [0 0 0];
if (nargin < 2)
    borderColor = [0 0 0];
end
for k =1:length(c);
    if (c(k).isvalid)
        max_sz = max(max_sz,size(c(k).vis));
    end
end
for k =1:length(c)
    if (c(k).isvalid)
        b = c(k).vis;
        sz = size(b);
        pad_size = max_sz-sz;
        b = padarray(b,pad_size(1:2),0,'post');
        if (nargin == 2)
            b = addBorder(b,2,borderColor);
        end
        m{k} = b;
    end
end
m = cat(1,m{:});

end

