function dist_min = compute_min_dist(distmat,train_bound,test_rect,train_regions,test_regions)
% function distmin = compute_min_dist(distmat,train_bound,test_rect,train_regions,test_regions)
%
% This function masks distance matrix distmat so that geometric matching
% between regions are considered.
%
% Copyright @ Chunhui Gu, April 2009

mask = false(size(distmat));

numtrainregions = length(train_regions);
numtestregions = length(test_regions);
train_ratio = zeros(numtrainregions,1);
test_ratio = zeros(numtestregions,1);
for trainId = 1:numtrainregions,
    
    [y,x] = find(train_regions{trainId}==true);
    train_rgsz = sqrt((max(y)-min(y))*(max(x)-min(x)));
    %train_rgsz = sqrt(length(x));
    if size(train_bound,1) == 1,
        I = 1;
    else
        centroid = [mean(y),mean(x)];
        bdcenters = train_bound(:,1:2)+0.5*train_bound(:,3:4);
        dd = sum((bdcenters - repmat(centroid,size(bdcenters,1),1)).^2,2);
        [ignore,I] = min(dd);
    end;
    train_bdsz = sqrt((train_bound(I,3)-train_bound(I,1))*(train_bound(I,4)-train_bound(I,2)));
    train_ratio(trainId) = train_rgsz/train_bdsz;
end;

for testId = 1:numtestregions,

    test_bdsz = sqrt(test_rect(3)*test_rect(4));
    [y,x] = find(test_regions{testId}==true);
    test_rgsz = sqrt((max(y)-min(y))*(max(x)-min(x)));
    %test_rgsz = sqrt(length(x));
    
    test_ratio(testId) = test_rgsz/test_bdsz;
end;

for trainId = 1:numtrainregions,
    for testId = 1:numtestregions,
        ratio = train_ratio(trainId) / test_ratio(testId);
        if ratio > 0.6 && ratio < 1/0.6,
            mask(trainId,testId) = true;
        end;
    end;
end;
        
%         subplot(1,2,1); imshow(train_regions{trainId});
%         rectangle('Position',[train_bound(:,1:2) train_bound(:,3:4)-train_bound(:,1:2)],'EdgeColor','r');
%         title('Exemplar image');
%         subplot(1,2,2); imshow(test_regions{testId});
%         rectangle('Position',test_rect,'EdgeColor','r');
%         title('Query image');
%         
%         ratio
%         keyboard;

dist_min = mask .* distmat + (~mask);