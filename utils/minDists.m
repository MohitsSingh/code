function [ inds ] = minDists( X,Y, nBatch)
%MINDISTS finds the nearest element in Y for every element in X, possibly
%in batches to increase memory efficiecy.
%
% dists = inf(size(X,2),1);
% if (nargin < 3)
%     nBatch = size(Y,2);
% end
% nBatch = min(size(Y,2),nBatch);
% inds = zeros(size(dists));

if (nargin < 3)
    nBatch = size(X,2);
end
nBatch = min(size(X,2),nBatch);

inds = {};
k = 0;
Y = Y';
for iStart = 1:nBatch:size(X,2)
%     fprintf(num2str(iStart));
    iEnd = min(iStart + nBatch-1,size(X,2));
    D = l2(single(X(:,iStart:iEnd)'),Y);
    [~,id] = min(D,[],2);
    k = k+1;
    inds{k} = id;
end


inds = cat(1,inds{:});

%
% for iStart = 1:nBatch:size(Y,2)
%     iEnd = min(iStart + nBatch-1,size(Y,2));
%     %     if (iEnd == iStart)
%     %         break;
%     %     end
%     D = l2(X',Y(:,iStart:iEnd)');
%     [d,id] = min(D,[],2);
%     id = id + iStart - 1;
%     %     nn = 0;
%     for k = 1:size(d,1)
%         if (d(k) < dists(k));
%             %             nn = nn+1;
%             inds(k) = id(k);
%             dists(k) = d(k);
%         end
%     end
%     %     disp(nn);
% end
% end

