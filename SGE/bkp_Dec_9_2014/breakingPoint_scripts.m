outDir = '~/storage/breaking_point_res';

testing = true;
params.scales = [3:25 30 35 40 50 60 80 100 150 200];
% params.scales = [3:20];
params.nTrials = 10;
addpath /home/amirro/code/breakingPoint;
voc_devkit_path = '/home/amirro/storage/data/voc07/VOCdevkit/VOCdevkit/VOCcode';
addpath(voc_devkit_path);
VOCinit;
params.classes = VOCopts.classes;
clear inputData; t = 0;
for iScale = 1:length(params.scales)
    for iClass = 1:length(params.classes)
        t = t+1;
        inputData(t).name = sprintf('cls_%s_scale_%03.0f',params.classes{iClass},params.scales(iScale));
        inputData(t).class = params.classes{iClass};
        inputData(t).scale = params.scales(iScale);
        inputData(t).nTrials = 10;
    end
end
testing = false;
run_all_3(inputData,outDir,'breakingPoint_extract_feats',testing,'mcluster03');