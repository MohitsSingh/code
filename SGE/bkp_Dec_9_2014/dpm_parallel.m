function dpm_parallel(baseDir,d,indRange,outDir)

% cd ~/code/mircs;
% load dpm_models/partModelsDPM.mat;
addpath('/home/amirro/code/3rdparty/uri');
addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
models_path = '/net/mraid11/export/data/amirro/root4sun/voc-release5/2012';
model_files = dir(fullfile(models_path,'*_final.mat'));
partModelsDPM = {};
for k = 1:length(model_files)
    load(fullfile(models_path,model_files(k).name));
    partModelsDPM{k} = model;
end
cd ~/code/3rdparty/voc-release5/

startup;
for k = 1:length(indRange)
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if filename %s exists :.... \n',resFileName);
    if (exist(resFileName,'file'))
        fprintf('already exists! skipping ... \n');
        continue;
    end
    currentID = d(indRange(k)).name;
    
    im = im2double(imread(imagePath));
    for iModel = 1:length(partModelsDPM)
        iModel
        model = partModelsDPM{iModel};
        dss = {};
        for iRot = 0:10:350
            curDS = detectRotated(im,model,-1,iRot);
            if (~isempty(curDS))
                curDS = [curDS,repmat(iRot,size(curDS,1),1)];
                dss{end+1} = curDS;
            end
        end
        ds = cat(1,dss{:});
        modelResults(iModel).class = model.class;
        modelResults(iModel).ds = ds;
    end
    save(resFileName,'modelResults');
    fprintf('done with image %s!\n',filename);
    fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end

