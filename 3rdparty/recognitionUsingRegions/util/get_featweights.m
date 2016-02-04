function weights = get_featweights(dist_map,labels,imgId,wtype,param)
% function weights = get_featweights(dist_map,labels,imgId,wtype,param)
%
% This function learns one weight on each feature based on its statistics
% in the discriminative task.
%
% Copyright @ Chunhui, April 2009

if nargin < 4,
    wtype = 'average';
end;
if nargin < 5,
    C = 1; precision = 0.001; numtimgs = 5;
else
    C = param.c; precision = param.prec; numtimgs = param.num;
end;

switch wtype
    case 'ranklearning'
        
        fdist = dist_map; fdist(:,imgId) = max(dist_map(:)) + 1;
        %%% choosing triplets
        [distdiff,count,sdpair] = get_distdiff(fdist,labels,numtimgs);
        ntriplets = size(distdiff,2);
        
        weights = ranklearning(distdiff,C,'l2',precision);
        fprintf('done %d(%d) triplets.\n',ntriplets,count);
        
    case 'ranklearning_region'
        
        regions = param.regions; frac = 0.01;
        fdist = dist_map; fdist(:,imgId) = max(dist_map(:)) + 1;
        
        rgsz = zeros(length(regions),1);
        for ww = 1:length(regions), rgsz(ww) = sum(sum(regions{ww})); end;
        ind = find(rgsz >= frac * max(rgsz));
        [distdiff,count,sdpair] = get_distdiff(fdist(ind,:),labels,numtimgs);
        ntriplets = size(distdiff,2);
        
        weights = zeros(length(regions),1);
        w = ranklearning(distdiff,C,'l2',precision);
        weights(ind) = w;
        fprintf('done %d(%d) triplets - ',ntriplets,count);        
    
    case 'todorovic'
        
        labels(imgId) = []; dist_map(:,imgId) = [];
        weights = recursivelearning(dist_map,labels,3);
        
    case 'average'
        
        weights = ones(size(dist_map,1),1);
        
    case 'region'
        
        weights = ones(size(dist_map,1),1);
        for ww = 1:length(regions), weights(ww) = sqrt(sum(sum(regions{ww}))); end;
end;

%%%%%%%%%%%%%%%%%%
function weights = recursivelearning(dist_map,labels,niters)

nfeats = size(dist_map,1);
weights = ones(nfeats,1); weights = weights / norm(weights,2);

dist_map_hit = dist_map(:,labels==1);
dist_map_miss = dist_map(:,labels==0);

sigma = 0.8;

for iter = 1:niters,
    dd_hit = weights'*dist_map_hit;
    dd_miss = weights'*dist_map_miss;
    
    prob_hit = exp(-dd_hit/sigma); prob_hit = prob_hit / sum(prob_hit);
    prob_miss = exp(-dd_miss/sigma); prob_miss = prob_miss / sum(prob_miss);
    
    [dummy, ind_hit] = min(dd_hit);
    [dummy, ind_miss] = min(dd_miss);
    
    weights = sum(dist_map_miss .* repmat(prob_miss,nfeats,1), 2) - ...
        sum(dist_map_hit .* repmat(prob_hit,nfeats,1), 2);
    weights = max(0, weights);
    weights = weights / sqrt(weights'*weights);
end;

%%%%%%%%%%%%%%%%%%
function [distdiff,count,sdpair] = get_distdiff_old(fdist,labels,K)

fndes = size(fdist,1);
distdiff = zeros(fndes,floor(K*K/4)*fndes);
sdpair = zeros(2,size(distdiff,2));
count = 0; kk = 0;
for ii = 1:fndes,
    [sindex,dindex] = selecttriplets(fdist(ii,:),labels,K);
    %%% constructing triplets
    for ss = 1:length(sindex),
        for dd = 1:length(dindex),
            if ~any(~any(sdpair(:,1:kk) - repmat([sindex(ss);dindex(dd)],1,kk))),
                kk = kk + 1;
                distdiff(:,kk) = fdist(:,dindex(dd))-fdist(:,sindex(ss));
                sdpair(:,kk) = [sindex(ss);dindex(dd)];
            else
                count = count + 1;
            end;
        end;
    end;
end;
distdiff = distdiff(:,1:kk);
sdpair = sdpair(:,1:kk);

%%%%%%%%%%%%%%%%%%
function [distdiff,count,sdpair] = get_distdiff(fdist,labels,K)

fndes = size(fdist,1);
sdpair = zeros(2,floor(K*K/4)*fndes);

kk = 0;
for ii = 1:fndes,
    %%% constructing triplets
    [sindex,dindex] = selecttriplets(fdist(ii,:),labels,K);
    ns = length(sindex); nd = length(dindex);
    sindex = repmat(sindex,1,nd);
    dindex = repmat(dindex,1,ns);
    sdpair(:,kk+1:kk+ns*nd) = [sindex;dindex];
    kk = kk + ns*nd;    
end;
sdpair = sdpair(:,1:kk);
% remove redundant ones
maxindex = size(fdist,2);
sdsingle = (sdpair(1,:)-1)*maxindex + sdpair(2,:);
[ignore,I] = unique(sdsingle);
count = length(sdsingle) - length(ignore);
sdpair = sdpair(:,I');

distdiff = double(fdist(:,sdpair(2,:))) - double(fdist(:,sdpair(1,:)));

%%%%%%%%%%%%%%%%%%
function [sindex,dindex] = selecttriplets(dist,label,K)

[dummy, I] = sort(dist);
if sum(label(I(1:K))) == 0,
    kk = K+1;
    while label(I(kk)) == 0,
        kk = kk + 1;
    end;
    sindex = I(kk);
    dindex = I(1:K);
elseif sum(label(I(1:K))) == K,
    kk = K+1;
    while label(I(kk)) == 1,
        kk = kk + 1;
    end;
    sindex = I(1:K);
    dindex = I(kk);
else
    vec = I(1:K);
    sindex = I(label(vec) == 1);
    dindex = I(label(vec) == 0);
end;