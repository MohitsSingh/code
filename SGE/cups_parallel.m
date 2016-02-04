function cups_parallel(baseDir,d,indRange,outDir)


% addpath('code/');
% addpath('code/pff_code/');
% addpath('code/pff_code/star-cascade');
% addpath('code/skin_based_detector');
% addpath('code/skin_based_detector/skindetector');
dpmdir = '/home/amirro/code/3rdparty/voc-release5';
cd(dpmdir);startup;


%for hand context model
load('/home/amirro/code/mircs/dpm_models/model_drinking_cups.mat');
for k = 1:length(indRange)
    i = indRange(k);
    imPath = fullfile(baseDir,d(i).name);
    fprintf(1,'Finding cups for image %s\n', imPath);
    resPath = fullfile(outDir,strrep(d(i).name,'.jpg','.mat'));
    if (exist(resPath,'file'))
        fprintf(1,'results for image %s already exist.\n', imPath);
        continue;
    end
    im = imread(imPath);
    disp('Running cup detector');
    [ds, bs] = process(im, model_drinking_cups,-1.1);
    %[boxes, boxes_r, bboxes] = my_imgdetect_r(im, shape_model, shape_model.thresh, fastflag);
%     if ~isempty(ds)
%         [ds, bs] = clipboxes(im, ds, bs);
%     end
%     
    boxes = single(ds);
    bboxes = single(bs);    
         
    save(resPath,'boxes','bboxes');
      
end

fprintf('\n\n\nFINISHED\n\n\n!\n');
end