
function binaryModels = learnBinaryModels(conf,train_ids,train_labels,groundTruth,learnParams)

if (~isfield(learnParams,'negImages'))
    negImages = train_ids(~train_labels);
else
    negImages = learnParams.negImages;
end
doHardNegativeMining = learnParams.doHardNegativeMining;
debugSuffix = learnParams.debugSuffix;
if (isfield(learnParams,'class_name'))
    class_name = learnParams.class_name;
else
    class_name = conf.classes{conf.class_subset};
end

featureExtractors = learnParams.featureExtractors;
partNames = learnParams.partNames;

binaryModels = struct('name',{},'model',{},'extractor',{},'id1',{},'id2',{});

conf.get_full_image = true;

compositeExtractor = CompositeFeatureExtractor(conf,featureExtractors);
gtParts = {groundTruth.name};

% create a new ground truth, with the intersection of different object
% types.

objInteractions = getObjectInteractions(groundTruth);

% co_occurence = sort([[objInteractions.id1];[objInteractions.id2]]',2);
 co_occurence = [[objInteractions.id1];[objInteractions.id2]]';

co_types = unique(co_occurence,'rows');

for iCo_type = 1:size(co_types)
    
    % for iPart = 1:length(partNames)
    %     for jPart = 1:length(partNames)
    %curSel = partIDS == iPart;
    %curSel = cellfun(@any,strfind(gtParts,partNames{iPart}));
    
    binaryModels(iCo_type).name = [partNames{co_types(iCo_type,1)} '_' partNames{co_types(iCo_type,2)}];
    binaryModels(iCo_type).id1 = co_types(iCo_type,1);
    binaryModels(iCo_type).id2 = co_types(iCo_type,2);
    curSel = co_occurence(:,1) == co_types(iCo_type,1) & co_occurence(:,2) == co_types(iCo_type,2);
    if (~nnz(curSel))
        continue;
    end
    binaryModels(iCo_type).models = trainModel(objInteractions(curSel),binaryModels(iCo_type).name,...
        compositeExtractor);
    binaryModels(iCo_type).extractor = compositeExtractor;
    %     if (~exist(extractorPath,'file'))
    %         save(extractorPath,'compositeExtractor');
    %     end
end

    function models = trainModel(groundTruth,modelName,featureExtractor)
        %         classifiers = [];
        % extract appearance features from each of the ground truth segments.
        
        debug_ = false;
        %         totalDescriptions = {};
        % classifiers = [];
%         for iExtractor = 1:length(featureExtractors)
            curExtractor = featureExtractor;%featureExtractors{iExtractor};
            classifierPath = fullfile(conf.cachedir,[class_name, '_', modelName '_' curExtractor.description  debugSuffix '_2.mat']);
            if (exist(classifierPath,'file'))
                load(classifierPath);
            else
                posImages = {};
                posRois = {};
                n = length(groundTruth);
                if debug_, n = 5; end
                
                for iObj = 1:n
                    currentID = groundTruth(iObj).sourceImage;
                    curImage = getImage(conf,currentID);
                    %R = roipoly(curImage,groundTruth(iObj).polygon.x,groundTruth(iObj).polygon.y);
                    R1 = roipoly(curImage,groundTruth(iObj).roi1(:,1),groundTruth(iObj).roi1(:,2));
                    R2 = roipoly(curImage,groundTruth(iObj).roi2(:,1),groundTruth(iObj).roi2(:,2));
                    posImages{iObj} = currentID;
                    posRois{iObj} = {R1 R2};
                end
                
                classifier = [];
                posFeats = getFeatures(curExtractor,posImages,posRois);
                if (isempty(posFeats))
                    classifier = [];
                else
                    
                    %
                    % train using an initial round of negatives and do hard-negative mining.                    
                    curNegImages = negImages(randperm(length(negImages)));
                    negFeats = {};
                    maxRounds = 5;
                    nImagesPerRound = 5;
                    maxNegatives = 10*size(posFeats,2);
                    if (debug_)
                        maxRounds = 1;
                        nImagesPerRound = conf.bow.imagesPerRound_debug;
                        maxNegatives = conf.bow.maxNegatives_debug;
                    end
                    for iRound = 1:maxRounds
                        if isempty(classifier)
                            imageSel = 1:nImagesPerRound;
                            % choose a few negatives from each image...
                            negFeats = getFeatures(curExtractor,curNegImages(imageSel),[],floor(maxNegatives/nImagesPerRound));
                            
                            classifier = train_classifier_pegasos(posFeats,negFeats);
                            
                            if (~doHardNegativeMining)
                                break;
                            end
                        else % mine hard negatives...
                            imageSel = max(imageSel):max(imageSel)+nImagesPerRound-1;
                            % extract descriptors from all regions in these images.
                            curNegatives = getFeatures(curExtractor,curNegImages(imageSel),[],floor(maxNegatives/nImagesPerRound));
                            curNegatives = [curNegatives,negFeats];
                            [~, scores] = classifier.test(curNegatives);
                            [~,iScore] = sort(scores,'descend');
                            iScore = iScore(1:min(maxNegatives,length(iScore)));
                            negFeats = curNegatives(:,iScore);
                            classifier = train_classifier_pegasos(posFeats,negFeats);
                        end
                    end
                end
                if ~debug_ ,save(classifierPath,'classifier'); end
                %
            end
            models = classifier;
%         end
    end
    function feats = getFeatures(curExtractor,images,rois,nFeatsPerImage)
        feats = {};
        if (nargin < 4)
            nFeatsPerImage = inf;
        end
        for k = 1:length(images)
            currentID = images{k};
            
            if (~isempty(rois))
                regions = rois{k};
                pairs = [1 2];
            else
                [regions,pairs] = getRegionPairSubset(conf,currentID,nFeatsPerImage);
            end
            
            x = curExtractor.extractFeatures(currentID,regions,pairs);
            x(:,isnan(sum(x))) = [];
            feats{end+1} = x;
        end
        empties = cellfun(@isempty,feats);
        feats = cat(2,feats{~empties});
    end
end



function objInteractions = getObjectInteractions(groundTruth)
objInteractions = struct('sourceImage',{},'id1',{},'id2',{},'roi1','roi2');
gtInds = zeros(size(groundTruth));
gtInds(1) = 1;
% split ground-truth according to source images.
count_ = 1;
for k = 2:length(groundTruth)
    if (~strcmp(groundTruth(k).sourceImage,groundTruth(k-1).sourceImage)) % still the same.
        count_ = count_+1;
    end
    gtInds(k) = count_;
end
u = unique(gtInds);
count_ = 1;
for k = 1:length(u)
    currentInds = find(gtInds==u(k));
    curObjects = groundTruth(currentInds);
    for i1 = 1:length(curObjects)
        [x1,y1] = poly2cw(curObjects(i1).polygon.x,curObjects(i1).polygon.y);
        for i2 = setdiff(1:length(curObjects),i1)%length(curObjects)
            [x2,y2] = poly2cw(curObjects(i2).polygon.x,curObjects(i2).polygon.y);
            objInteractions(count_).sourceImage = groundTruth(currentInds(1)).sourceImage;
            objInteractions(count_).id1 = curObjects(i1).partID;
            objInteractions(count_).id2 = curObjects(i2).partID;
            objInteractions(count_).roi1 = [x1 y1];
            objInteractions(count_).roi2 = [x2 y2];
            count_=count_+1;
        end
    end
end


end


