function [data,inds] = splitToRanges(d,nSplits)

inds = {};
if (nargin < 2)
    nSplits = 100;
end
% d = d(randperm(length(d)));
p = randperm(length(d));
m = mod(p,nSplits);

for k = 0:nSplits-1
    %    inds{k+1} = d(find(m==k));
    inds{k+1} = find(m==k);
    data{k+1} = d(inds{k+1});
end
goods = cellfun(@(x) ~isempty(x),inds);
inds = inds(goods);
data = data(goods);