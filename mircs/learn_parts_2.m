function [models_parts,models_links,models_shape] = learn_parts_2(all_pos_feats,all_neg_feats,nParts,lambda)

if nargin < 4
    lambda = [1 .1 .01 .001];
end

models_parts = struct('w',{},'b',{},'valid',{});
models_links = struct('w',{},'b',{},'valid',{});
models_shape = struct('w',{},'b',{},'valid',{});

for iPart = 1:nParts
    models_parts(iPart) = getFeatsAndTrain(all_pos_feats,all_neg_feats,'partFeats',iPart,lambda);
end

for iPart = 1:nParts
    models_shape(iPart) = getFeatsAndTrain(all_pos_feats,all_neg_feats,'shapeFeats',iPart,lambda);
end

for iLink = 1:nParts-1
    models_links(iLink) = getFeatsAndTrain(all_pos_feats,all_neg_feats,'intFeats',iLink,lambda);
end

function res = getFeatsAndTrain(all_pos_feats,all_neg_feats,featName,iPart,lambda)

res = struct('w',[],'b',[],'valid',false);

pos_feats = getFeats(all_pos_feats,featName,iPart);
if isempty(pos_feats)
    return;
end
neg_feats = getFeats(all_neg_feats,featName,iPart);
[x,y] = featsToLabels(pos_feats,neg_feats);
p = Pegasos(x,y,'lambda',lambda,'foldNum',5,'bias',1);
res = struct('w',p.w(1:end-1),'b',p.w(end),'valid',true);

function feats = getFeats(featStruct,featName,iPart)
feats = {};

if checkEmpty([featStruct.(featName)])
    return
end

for t = 1:length(featStruct)
    f = featStruct(t).(featName);
    if (isempty(f))
        fprintf(['warning : found empty features in ' num2str(t) ' - skipping']);
        continue;
    end
    f = cellfun3(@(x) x(:,iPart),f,2);
    feats{t} = f;
end
feats = cat(2,feats{:});

function res = checkEmpty(x)
res = all(cellfun3(@isempty,x,2));


