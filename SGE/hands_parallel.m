function hands_parallel(baseDir,d,indRange,outDir)

handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
cd(handDetectionBaseDir);

addpath('code/');
addpath('code/pff_code/');
addpath('code/pff_code/star-cascade');
addpath('code/skin_based_detector');
addpath('code/skin_based_detector/skindetector');

pca = 5;
thresh = -1;

fastflag = 0; % for faster cascaded version, else set it to 0.

%for hand shape model
%load('trained_models/hand_shape_final.mat');
load('trained_models/hands_final.mat'); % from ita
model.bboxpred = [];
if(fastflag)
    csc_shape_model = cascade_model(model,'shape',pca,thresh);
    shape_model = csc_shape_model;
else
    shape_model = model;
end

%for hand context model
load('trained_models/context_final.mat');
model.bboxpred = [];
if(fastflag)
    context_model = cascade_model(model,'shape',pca,thresh);
else
    context_model = model;
end
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
    [boxes, boxes_r, bboxes] = my_imgdetect_r(im, shape_model, shape_model.thresh, fastflag);
    if ~isempty(boxes)
        [boxes, bboxes] = clipboxes(im, boxes, bboxes);
    end
    
    boxes  = esvm_nms(boxes,.5);
%   
%     disp('Running hand context detector');
%     [boxes, boxes_r, bboxes] = my_imgdetect_r(im, context_model, context_model.thresh, fastflag);
%     if ~isempty(boxes)
%         [boxes, bboxes] = clipboxes(im, boxes, bboxes);
%     end
%     
%     context.boxes = boxes;
%     context.bboxes = bboxes;
%     context.boxes_r = boxes_r;
    
    save(resPath, 'boxes');
    
    %     disp('Getting skin regions');
    %     getSkinRegions(im, i);
    %
    %     disp('Running skin based detector');
    %     getSkinBoxes(im, i);
end

fprintf('\n\n\nFINISHED\n\n\n!\n');
end