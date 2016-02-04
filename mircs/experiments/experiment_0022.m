%%% Experiment 22 %%%%
% Extract fragments from many objects in order to detect them when they are
% occluded as well. 

% first download some stuff from imagenet.
p = pwd;
cd /home/amirro/code/3rdparty/ImageNetToolboxV0.3/;


cd /home/amirro/storage/root4sun/VOC2012/VOCdevkit;
addpath('VOCcode/');

VOCinit;

addpath(genpath('/net/mraid11/export/data/amirro/root4sun/voc-release5'));
conf = voc_config;
cls = 'coffee cup';

[pos, neg, impos] = pascal_data(cls, conf.pascal.year);

for k = 1:length(impos)
    if (impos(k).flip)
        continue;
    end
    I = imread(impos(k).im);
    
    clf; showboxes(I,impos(k).boxes);
    pause;
    
end
addpath('/home/amirro/storage/datasets/image_net/auto_annotation');

data = read_boxes('boxes.bin');
save boxes.mat data
sub_data = get_synset_images(data,'n03147509'); % bottle

% ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');
% cls = 'bottle';
% for k = 1:length(ids)
%     rec           = PASreadrecord(sprintf(VOCopts.annopath, ids{k}));
%     clsinds       = strmatch(cls, {rec.objects(:).class}, 'exact');
%     diff          = [rec.objects(clsinds).difficult];
%     clsinds(diff) = [];
%     count         = length(clsinds(:));
% end