function [ f,d] = stack_features(f,d)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

pmax = max(f(1:2,:)',[],1);
scales = f(4,:);
u_scales = unique(scales);
n_scales = length(u_scales);
height = pmax(2);
width = pmax(1);
counts = zeros(height,width);
z_feats = cell(height,width,n_scales);
[lia,locb] = ismember(scales,u_scales);


for t = 1:size(f,2)    
    ii = f(2,t);jj=f(1,t);
    counts(ii,jj)= counts(ii,jj)+1;
    z_feats{ii,jj,locb(t)} = d(:,t);
end

goods = counts==n_scales;
[yy,xx] = find(goods);
f = [xx';yy'];
dd = {};
for t = 1:length(xx)
    dd{t} = col(cat(3,z_feats{yy(t),xx(t),:}));
end
d = cat(2,dd{:});
%z_feats = z_feats(goods);