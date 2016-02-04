function [radii,sums,dists] = cummass(P,centerPt)
[xx,yy] = meshgrid(1:size(P,2),1:size(P,1));
% xx = xx-centerPt(1);
% yy = yy-centerPt(2);
dists = ((xx-centerPt(1)).^2 +(yy-centerPt(2)).^2).^.5;

min_rad = 1;
max_rad = size(P,1);
%radii = [linspace(min_rad,max_rad,50) inf];
radii = [0 min_rad:5:max_rad,50 inf];
sums = zeros(size(radii));
V = zeros(size(P));
for t = 2:length(sums)
    sums(t) = sum(P(dists < radii(t)));
    if (t < length(sums))
        V(dists >= radii(t-1) & dists<radii(t)) = sums(t);
    end
end
V = V/max(V(:));
sums = sums/sums(end);
% figure,plot(radii,sums);
%[n,bins] = hist(dists(:),50);
% nr = 50;
% nw = 1;
% logarr = logsample(dists, 1, size(dists,1), centerPt(1), centerPt(2), nr, nw);
% f = find(cumsum(logarr)>=.8*sum(logarr));
% U = logsampback(logarr, 1, size(dists,1));
% figure,plot(logarr);
% figure,imagesc2(logarr);
end
