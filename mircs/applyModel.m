function [regionConfs] =  applyModel(conf,currentID,models,outDir)
if (nargin <4)
    outDir = '~/storage/res_s40';
end
resPath = fullfile(outDir,strrep(currentID,'.jpg','.mat'));
if (exist(resPath,'file'))
    load(resPath);
    return
end
regionConfs = struct('score',{},'pred',{});
for iModel = 1:length(models)
    m = models(iModel);
    feats = m.extractor.extractFeatures(currentID);
    %     w = m.models.w;
    [regionConfs(iModel).pred, regionConfs(iModel).score] = m.models.test(feats);
    %     save(resPath,'regionConfs');
end