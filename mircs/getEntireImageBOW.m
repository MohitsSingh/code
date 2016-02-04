function bow = getEntireImageBOW(conf,ids,featureExtractor)
    bow = {};
    for k = 1:length(ids)
        k
        bow{k} = featureExtractor.extractFeatures(ids{k},-1);
    end
    bow = cat(2,bow{:});
end
