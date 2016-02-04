function batches = batchify(n,nBatches)
if nargin < 2
    nBatches = 10;
end
batches = {};
batchSize = floor(n/nBatches);
for starts = 1:batchSize:n
    curStart = starts;
    curEnd = min(curStart+batchSize-1,n);
    batches{end+1} = curStart:curEnd;
end


end