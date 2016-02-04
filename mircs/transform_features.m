function res = transform_features(inputFeats,featParams,valids)
res = {};
for iFeat = 1:length(inputFeats)
    if featParams.normalize_each
        res{iFeat} = normalize_vec(inputFeats(iFeat).feats);
    else
        res{iFeat} = inputFeats(iFeat).feats;
    end
end
res = cat(1,res{:});
if featParams.normalize_all
    res = normalize_vec(res);
end


end