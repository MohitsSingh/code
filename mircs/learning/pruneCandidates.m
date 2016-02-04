function candidates = pruneCandidates(imgData, prevPhase, prevCandidates,imgInd)
%PRUNECANDIDATES(imgData,prevPhase,prevPhase) removes candidates with low
%scores 

    % select a classifier which does not contain this image index
    classifiers = prevPhase.classifiers;
    T = [];    
    for t = 1:length(classifiers)
        curClassifier = classifiers(t);
%         if ~curClassifier.fold(imgInd)
        if ~ismember(imgInd,curClassifier.fold)
            T = t;
            break;
        end
    end
    if isempty(T)
        error('prune candidates: could not find a classifier which was not trained on current image');
    end
        
    
    featureExtractor = prevPhase.featureExtractor;
    features = featureExtractor.extractFeaturesMulti_mask(imgData.I_sub,prevCandidates);
    curClassifier = classifiers(T);
    [ws bs] = get_w_from_classifiers(curClassifier);
    scores = bsxfun(@plus,ws'*features,bs);    
    [r,ir] = sort(scores,'descend');
    nToKeep = prevPhase.nToKeep;
    ir = ir(1:min(length(ir),nToKeep)); % select only the top 10....
    candidates = prevCandidates(ir);

end

