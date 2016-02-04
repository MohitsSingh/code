function [res] = inImageBounds( bbox,pt)
%INIMAGEBOUNDS check if xy coordinates are in image of size sz
%   Detailed explanation goes here
if (numel(bbox)>4)
    bbox = size2(bbox);
end
if (size(bbox)<4)
    bbox = [1 1 bbox(2) bbox(1)];
end

res = true(size(pt,1),1);
for u = 1:size(pt,2)/2    
    res = res & inBox(bbox,pt(:,(u-1)*2+1:u*2));
end
%     res = pt(:,1) >=1 & pt(:,2) >=1 & pt(:,1)<=sz(2) & pt(:,2) <=sz(1);

end

