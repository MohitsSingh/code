function [feats,valids] = extractFeatures_(featureExtractor,ims,valids)
feats = {};
if (~iscell(ims))
    ims = {ims};
end
if (nargin < 3)
    valids = true(size(ims));
end

if (none(valids))
    error('no valid images...!');
end

ticId = ticStatus([],.5,.1);
for k = 1:length(ims)
    if (~valids(k)), continue; end
    tocStatus( ticId, k/length(ims));
    %     100*k/length(ims)
    feats{k} = col(featureExtractor.extractFeatures(ims{k}));
end

f_valid = find(valids,1,'first');
for k = 1:length(ims)
    if (valids(k)), continue; end
    feats{k} = zeros(size(feats{f_valid}));
end
feats = cat(2,feats{:});
valids = all(~isnan(feats));

end