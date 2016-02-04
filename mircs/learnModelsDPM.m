function partModelsDPM = learnModelsDPM(conf,train_ids,train_labels,groundTruth,partNames,false_images)
baseDir = 'dpm_models';
mkdir(baseDir);
if (~exist('false_images','var'))
    false_ids = train_ids(~train_labels);
else
    false_ids = false_images;
end



gtParts = {groundTruth.name};

for iPart  = 1:length(partNames)
    

    cls = [conf.classes{conf.class_subset} '_' partNames{iPart}];
    partPath = fullfile(baseDir,[cls '.mat']);
    if (exist(partPath,'file'))
        continue;
    end
    %currentImages = find([groundTruth.partID] == iPart);
    currentImages = find(cellfun(@any,strfind(gtParts,partNames{iPart})));
    true_ids = {};
    posBoxes = {};
    for k = 1:length(currentImages)
        true_ids{k} = groundTruth(currentImages(k)).sourceImage;
        
        posBoxes{k} = [ pts2Box([groundTruth(currentImages(k)).polygon.x,...
            groundTruth(currentImages(k)).polygon.y]) groundTruth(currentImages(k)).curTheta];
    end
    posBoxes = cat(1,posBoxes{:});
    
    trainSet = prepareForDPM(conf,true_ids,false_ids,posBoxes);
    
    n = 1; % number of subclasses
    valSet = [];
    partModelsDPM{iPart} = runDPMLearning(cls, n, trainSet, valSet);
end
