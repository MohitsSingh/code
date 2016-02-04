function bowHists = getBowHists(conf,imageSet,dict)
bowHists = zeros(size(dict,2),length(imageSet));
for k = 1:length(imageSet)
    if (mod(k,100))
        disp(k)
    end
    [bowFeatures] = getBows(conf.VOCopts,imageSet(k),dict,'data/bow');
    bowFeatures = bowFeatures{1};
    bowHists(:,k) = hist(single(bowFeatures),1:size(dict,2));
end
