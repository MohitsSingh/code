function dpm_subclass_parallel(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;
config;
load ~/storage/misc/imageData_new;
load(fullfile(conf.cachedir,'obj_dpm.mat')) % models
addpath(genpath('/home/amirro/code/3rdparty/voc-release5'));

for k = 1:length(indRange)
    k
    curImageData = newImageData(indRange(k));
    if (curImageData.faceScore < -.6)
        continue;
    end
    resPath = fullfile(outDir,strrep(curImageData.imageID,'.jpg','.mat'));
    % run o
    %     if (~exist(resPath,'file'))
    res = struct('class',{},'boxes',{},'fullBoxes',{});
    for iModel = 1:length(models)
        res(iModel).class = models{iModel}.class;
        res(iModel).boxes = getResponseMap(conf,curImageData,models{iModel});
        %             res(iModel).fullBoxes = getResponseMap(conf,curImageData,models{iModel},'full');
    end
    save(resPath,'res');
    %     end
    fprintf('done with image %s!\n',curImageData.imageID);
end
fprintf('***********************************\n************FINISHED***************\n***********************************\n');
end

