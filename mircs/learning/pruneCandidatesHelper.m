function curClassifier = pruneCandidatesHelper(prevPhase,imgInd)
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
    curClassifier = classifiers(T);    
end