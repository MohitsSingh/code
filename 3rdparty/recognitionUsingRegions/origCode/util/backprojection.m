function [BB, I] = backprojection(r1,bound,r2)
% function [BB, I] = backprojection(r1,bound,r2)
%
% This function back-projects the input bounding box "bound" based on
% transformation between regions r1 and r2. The transformation is allowed
% to vary to translation and scale in both x- and y-directions.
%
% Copyright @ Chunhui Gu, April 2009

nboxes = size(bound,1);
int = zeros(nboxes,1);
for rr = 1:nboxes,
    mask = false(size(r1));
    mask(bound(rr,2):bound(rr,4),bound(rr,1):bound(rr,3)) = true;
    int(rr) = sum(sum(mask & r1));
end;
[ignore,I] = max(int);
bound = bound(I,:);

box.asp = (bound(4)-bound(2))/(bound(3)-bound(1)); % y/x
box.scale = bound(3) - bound(1);
box.x = 0.5*(bound(1)+bound(3));
box.y = 0.5*(bound(2)+bound(4));

%r1stats = regionprops(double(r1),'basic');
%r2stats = regionprops(double(r2),'basic');
[y,x] = find(r1==true);
r1stats.BoundingBox = [min(x)-0.5 min(y)-0.5 max(x)-min(x)+1 max(y)-min(y)+1];
r1stats.Centroid = [mean(x) mean(y)];

[y,x] = find(r2==true);
r2stats.BoundingBox = [min(x)-0.5 min(y)-0.5 max(x)-min(x)+1 max(y)-min(y)+1];
r2stats.Centroid = [mean(x) mean(y)];

box_new.asp = box.asp;
box_new.scale = box.scale * sqrt(r2stats.BoundingBox(3)*r2stats.BoundingBox(4)) / sqrt(r1stats.BoundingBox(3)*r1stats.BoundingBox(4));
box_new.x = r2stats.Centroid(1) + box_new.scale/box.scale * (box.x-r1stats.Centroid(1));
box_new.y = r2stats.Centroid(2) + box_new.scale/box.scale * (box.y-r1stats.Centroid(2));

W = round(box_new.scale);
H = round(W*box_new.asp);
X0 = round(box_new.x - 0.5*W);
Y0 = round(box_new.y - 0.5*H);

BB = [X0,Y0,X0+W,Y0+H];