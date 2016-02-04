function get_binary_feats_parallel(baseDir,d,indRange,outDir)



cd /home/amirro/code/mircs;
initpath;
config;
initBinaryModels;
% initpath;
% config;
% conf.get_full_image = true;
% r = RelativeShapeFeatureExtractor(conf);
% r.doPostProcess = true;
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    if (exist(resFileName,'file'))
        fprintf('results exist. skipping\n');
        continue;
    else
        fprintf('calculating for image %s...\n',currentID);
    end
    %   [regionConfs] = applyModel(conf,currentID,partModels);
    [regions,regionOVP,G] = getRegions(conf,currentID);
    [ii,jj] = find(G);
    %   relativeShapes = r.extractFeatures(regions,[ii jj]);
    pairs = [ii jj];
    feats = binaryModels(1).extractor.extractFeatures(currentID, regions, pairs);
    scores = {};
    for iModel = 1:length(binaryModels)
        if (~isempty(binaryModels(iModel).models))
            [ignore_, scores{iModel}] = binaryModels(iModel).models.test(feats);
        end
    end
    save(resFileName,'pairs','scores');
    %     fprintf('\tdone!\n');
end
% fprintf('\n\n\nFINISHED\n\n\n!\n');
end