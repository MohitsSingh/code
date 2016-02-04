function res = applyToImageSet(imgSet,classifiers,params)
% apply the learned classifier to the set of images in imgSet,
% possibly retaining only a subset of the results for each image.
%%  apply on validation images, to retain only top-scoring regions
% labels = {};
% decision_values = {};
% ovps = {};
% imgInds = {};
res = struct('labels',{},'decision_values',{},'ovps',{},'imgInd',{});
id = ticStatus( 'computing on image set...', .5, .5, true);
for k = 1:length(imgSet)
    
    resPath = j2m(params.dataDir,imgSet(k));
    imgSet(k).imageID
    res(k).imgInd = imgSet(k).imgIndex;
    if (~exist(resPath,'file'))
        res(k).decision_values = -inf(4,1);
        continue
    end
    % if (fra_db(u).classID~=9),continue,end
    try 
        L = load(resPath,'feats','moreData');
    catch e
        res(k).decision_values = -inf(4,1);continue
    end
    if (isempty(L.feats))
        res(k).decision_values = -inf(4,1);continue
    end
    [all_labels,currentFeatures,ovp] = collectFeatures(L,params.features);
    res(k).labels = all_labels;
    decision_values = NaN(length(classifiers),length(all_labels));
    for iClassifier = 1:length(classifiers)
        decision_values(iClassifier,:) = apply_region_classifier(classifiers(iClassifier), currentFeatures,params);
    end
    res(k).decision_values = decision_values;
    res(k).ovps = ovp;
    k/length(imgSet)
%     tocStatus(id,k/length(imgSet));
end
