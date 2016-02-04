function drink_parallel(baseDir,d,indRange,outDir)

detectionBaseDir = '/home/amirro/code/3rdparty/voc-release5';
cd(detectionBaseDir );
startup;

% 
% % addpath('code/');
% addpath('code/pff_code/');
% addpath('code/pff_code/star-cascade');

pca = 5;
thresh = -1;
load('/home/amirro/code/mircs/dpm_models/bottle.mat');
load('/home/amirro/code/mircs/dpm_models/cup.mat');

bottleModel = partModelsDPM_bottle{1};
cupModel = partModelsDPM_cup{1};
models = [bottleModel,cupModel];
modelNames = {'bottle','cup'};
fastflag = 0; % for faster cascaded version, else set it to 0.

%for hand context model
for k = 1:length(indRange)
    i = indRange(k);
    imPath = fullfile(baseDir,d(i).name);
    fprintf(1,'Generating hypotheses for image %s\n', imPath);
    resPath = fullfile(outDir,strrep(d(i).name,'.jpg','.mat'));
    if (exist(resPath,'file'))
        fprintf(1,'results for image %s already exist.\n', imPath);
        continue;
    end
    im = imread(imPath);
    disp('Running hand shape detector');
    
    for iModel = 1:length(models)
        shape_model = models(iModel);
        [boxes, boxes_r, bboxes] = my_imgdetect_r(im, shape_model, shape_model.thresh , fastflag);
        if ~isempty(boxes)
            [boxes, bboxes] = clipboxes(im, boxes, bboxes);
        end
        
                        
        shape(iModel).boxes = boxes;
%         shape(iModel).bboxes = bboxes(top,:);
%         shape(iModel).boxes_r = boxes_r(top,:);
        shape(iModel).name = modelNames{iModel};        
    end
    
    save(resPath,'shape');
    
    fprintf('\n\n\nFINISHED\n\n\n!\n');
end