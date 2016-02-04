beyond_hnm_path = '/home/amirro/code/3rdparty/beyond_hnm/';
addpath(genpath(beyond_hnm_path));
annotationDir = '/home/amirro/storage/datasets/image_net/Annotation';
addpath('/home/amirro/code/3rdparty/ImageNetToolboxV0.3');
addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
load synsets;
ff = findIsAnnotated(synsets,annotationDir);
neededClasses = {synsets(ff).wnid};
load recs_exist;
VOCPath = '/home/amirro/storage/datasets/image_net/VOCdevkit';
addpath(genpath(VOCPath));
VOCinit;
VOCopts.imgpath = '/home/amirro/storage/datasets/image_net/images_unpacked/%s.JPEG';
VOCopts.annopath = '/home/amirro/storage/datasets/image_net/Annotation/unpacked/%s.xml';
global extra;
extra.suffix = '_s40_neg';
extra.recs = recs;
extra.classes = neededClasses;
extra.VOCopts = VOCopts;

%negativeDir = '~/storage/copydays_original/';
% override the VOCopts to match the imagenet data.
run_circulant_custom
wnids = {synsets.wnid};
% match each classifier with a name
for k = 1:length(classifiers)
    f = find(cellfun(@any,strfind(wnids,classifiers(k).class)));
    classifiers(k).name = synsets(f).name;    
end


