function allres = detect_dpm_parallel(imagePaths,inds,jobID,outDir,extraInfo,job_suffix)
cd ~/code/mircs/;
conf = extraInfo.conf;
path(extraInfo.path);
models = extraInfo.models;
newImageData = extraInfo.newImageData;

if (~isfield(extraInfo,'runMode'))
    runMode = 'full';
else
    runMode = extraInfo.runMode;
end

if (~isfield(extraInfo,'minFaceScore'))
    minFaceScore = -.6;
else
    minFaceScore = extraInfo.minFaceScore;
end
allres = {};
for k = 1:length(inds)
    k
    res = struct('class',{},'boxes',{});
    curImageData = newImageData(inds(k));
    if (curImageData.faceScore >= minFaceScore)
        resPath = fullfile(outDir,strrep(curImageData.imageID,'.jpg',['_' job_suffix '.mat']));
        if (~exist(resPath,'file'))
            for iModel = 1:length(models)
                res(iModel).class = models{iModel}.class;
                res(iModel).boxes = getResponseMap(conf,curImageData,models{iModel},runMode);
            end
            save(resPath,'res');
        else
            load(resPath);
        end
    else
        for iModel = 1:length(models)
            res(iModel).class = models{iModel}.class;
            res(iModel).boxes = [];
        end
    end
    %     end
    allres{k} = res;
    fprintf('done with image %s!\n',curImageData.imageID);
end
fprintf('***********************************\n************FINISHED***************\n***********************************\n');

res = allres;
resFileName = fullfile(outDir,sprintf('job_%05.0f_%s_agg.mat',jobID,job_suffix));
save(resFileName,'res','inds','jobID');