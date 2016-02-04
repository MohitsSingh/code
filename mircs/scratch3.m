% load ~/storage/misc/mouth_images2.mat

initpath;
config;
conf.get_full_image = true;
%     [learnParams,conf] = getDefaultLearningParams(conf,1024);
load fra_db.mat;
all_class_names = {fra_db.class};
class_labels = [fra_db.classID];
classes = unique(class_labels);
[lia,lib] = ismember(classes,class_labels);
classNames = all_class_names(lib);

addpath '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';install;


I = mouth_images{101};
I = imResample(I,[64 64],'bilinear');
[candidates, ucm] = im2mcg(I,'fast',false); % warning, using fast segmentation...

x2(I);
x2(ucm)