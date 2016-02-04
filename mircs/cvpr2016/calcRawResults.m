function res = calcRawResults(gt_maps,test_maps,map_type,train_params)
n = 0;

res = struct('class_id',{},'map_type',{},'recall',{},'precision',{},'info',{});

for iClass = 1:length(train_params.classes)
    iClass
    DDD = 2;
    b = cellfun3(@(x) 2*(col(x(1:DDD:end,1:DDD:end)) == iClass+2)-1,gt_maps,1);
    scores = cellfun3(@(x) col(x(1:DDD:end,1:DDD:end,iClass+3)),test_maps);
    [recall,precision,info] = vl_pr(b,scores);
    recall = recall(1:10:end);
    precision = precision(1:10:end);
    n = n+1;
    res(n) = struct('class_id',iClass,'map_type',map_type,'recall',recall,'precision',...
        precision,'info',info);
end

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


end

