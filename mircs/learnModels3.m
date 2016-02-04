
function partModels = learnModels3(conf,groundTruth,learnParams,imageData)

 
faceBoxes = imageData.train.faceBoxes;
imageIDs = imageData.train.imageIDs;
posImages = imageIDs(imageData.train.labels);
negFaceBoxes = faceBoxes(~imageData.train.labels,:);
negImages = imageIDs(~imageData.train.labels);

% negImages = learnParams.negImages;
doHardNegativeMining = learnParams.doHardNegativeMining;
debugSuffix = learnParams.debugSuffix;
if (isfield(learnParams,'class_name'))
    class_name = learnParams.class_name;
else
    class_name = conf.classes{conf.class_subset};
end

featureExtractors = learnParams.featureExtractors;
partNames = learnParams.partNames;
useRealGTSegments = learnParams.useRealGTSegments;
partModels = struct('name',{},'model',{},'extractor',{});
exclusiveLabels = learnParams.exclusiveLabels;
conf.get_full_image = true;
nNegativesPerPositive = learnParams.nNegativesPerPositive;
nNegativeMiningRounds = learnParams.nNegativeMiningRounds;
compositeExtractor = CompositeFeatureExtractor(conf,featureExtractors);
compositeExtractor.doPostProcess = false;
gtParts = {groundTruth.name};

for iPart = 1:length(partNames)
    curSel = cellfun(@any,strfind(gtParts,partNames{iPart}));
    partModels(iPart).name = partNames{iPart};
    otherGT = groundTruth(~curSel);
    partModels(iPart).models = trainModel(groundTruth(curSel),partModels(iPart).name,...
        {compositeExtractor});
    partModels(iPart).extractor = compositeExtractor;
    
    
end
    function classifiers = trainModel(groundTruth,modelName,featureExtractors)
        
        % extract appearance features from each of the ground truth segments.
        
        debug_ = false;
        %         totalDescriptions = {};
        % classifiers = [];
        for iExtractor = 1:length(featureExtractors)
            curExtractor = featureExtractors{iExtractor};
            classifierPath = fullfile(conf.cachedir,[class_name, '_', modelName '_' curExtractor.description  debugSuffix '.mat']);
            if (exist(classifierPath,'file'))
                load(classifierPath);
            else
                posImages = {};
                posRois = {};
                n = length(groundTruth);
                if debug_, n = 5; end
                
                for iObj = 1:n
                    iObj
                    currentID = groundTruth(iObj).sourceImage;
                    curImage = getImage(conf,currentID);
                    R = roipoly(curImage,groundTruth(iObj).polygon.x,groundTruth(iObj).polygon.y);
                    
                    if (exclusiveLabels)
                        otherObjects =  find(cellfun(@any,(strfind({otherGT.sourceImage},currentID))));
                        if (~isempty(otherObjects))
                            RR = zeros(dsize(curImage,1:2));
                            for iOther = 1:length(otherObjects)
                                g = otherGT(otherObjects(iOther)).polygon;
                                RR = RR | roipoly(curImage,g.x,g.y);
                            end
                        end
                        R = R & ~RR;
                    end
                    
                    if (useRealGTSegments)
                        regions = getRegions(conf,currentID,false);
                        [ovp,ints,areas] = boxRegionOverlap(R,regions);
                        [~,iovp] = max(ovp);
                        posRois{iObj} = regions{iovp};
                        %                         displayRegions(curImage,{ posRois{iObj}});
                    else
                        posRois{iObj} = R;
                    end
                    
                    
                    posImages{iObj} = currentID;
                    
                end
                
                posFeats = getFeatures(curExtractor,posImages,posRois);
                
                %
                % train using an initial round of negatives and do hard-negative mining.
                classifier = [];
%                 curNegImages = negImages(randperm(length(negImages)));
                negImageSel = randperm(length(negImages));
                negFeats = {};
                nImagesPerRound = 5; % TODO. was 5
                maxNegatives = nNegativesPerPositive*size(posFeats,2);
                if (debug_)
                    maxRounds = 1;
                    nImagesPerRound = conf.bow.imagesPerRound_debug;
                    maxNegatives = conf.bow.maxNegatives_debug;
                end
                for iRound = 1:nNegativeMiningRounds
                    if isempty(classifier)
                        imageSel = 1:nImagesPerRound;
                        % choose a few negatives from each image...                                                                        
                        
                        negFeats = getFeatures(curExtractor,negImages(negImageSel(imageSel)),negFaceBoxes(negImageSel(imageSel),:),floor(maxNegatives/nImagesPerRound));
                        
                        classifier = train_classifier_pegasos(posFeats,negFeats,learnParams);
                        
                        if (~doHardNegativeMining)
                            break;
                        end
                    else % mine hard negatives...
                        imageSel = max(imageSel)+1:max(imageSel)+nImagesPerRound;
                        % extract descriptors from all regions in these images.
                        %curNegatives = getFeatures(curExtractor,curNegImages(imageSel),[]);
                        curNegatives = getFeatures(curExtractor,...
                            negImages(negImageSel(imageSel)),...
                            negFaceBoxes(negImageSel(imageSel),:));                       
                        curNegatives = [curNegatives,negFeats];
                        [~, scores] = classifier.test(curNegatives);
                        [~,iScore] = sort(scores,'descend');
                        iScore = iScore(1:min(maxNegatives,length(iScore)));
                        negFeats = curNegatives(:,iScore);
                        classifier = train_classifier_pegasos(posFeats,negFeats,learnParams);
                    end
                end
                if ~debug_ ,save(classifierPath,'classifier'); end
                %
            end
            classifiers(iExtractor) = classifier;
        end
    end

    function feats = getFeatures(curExtractor,images,rois,nFeatsPerImage)
        feats = {};
        if (nargin < 4)
            nFeatsPerImage = inf;
        end
        for k = 1:length(images)
            k
            currentID = images{k};
            if (~isempty(rois))
                if (iscell(rois)) 
                    regions = rois(k);
                else % bounding box...
                    regions = getRegions(conf,currentID,false);
                    [ovp,ints,areas] = boxRegionOverlap(rois(k,:),regions);
                    regions = regions(ovp>0);
                end
            else
                regions = getRegionSubset(conf,currentID,nFeatsPerImage);
            end
            
            x = curExtractor.extractFeatures(currentID,regions);
            x(:,isnan(sum(x))) = [];
            feats{end+1} = x;
        end
        feats = cat(2,feats{:});
    end
end

function classifier = train_classifier_pegasos(posFeats,negFeats,learnParams)
if (learnParams.balanceDatasets)
    posFeats = repmat(posFeats,[1 round(size(negFeats,2)/size(posFeats,2))]);
end
x = [posFeats,negFeats];
y = zeros(size(x,2),1);
y(1:size(posFeats,2)) = 1;
y(size(posFeats,2)+1:end) = -1;
classifier = Pegasos(x(:,:),y);
% classifier = Piotr_boosting(x(:,:),y);
end
