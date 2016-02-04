function [ids,labels,all_labels] = getImageSet(conf,setType,ratio,offset,classes)
%GETTRAINIMAGES retrieves the ids of the image set define by setType
%(either 'train' or 'test' and returns a label vector specifying
% which ids contain the class in conf.clas_subset (currently 1 class is
% supported, so this is a 1.vs.all classiciation)
%   Detailed explanation goes here
z = 0;
if (nargin < 4)
    offset = 0;
end
if (nargin < 3)
    ratio = 1;
end

if (~strcmp(setType,'train') && ~strcmp(setType,'test'))
    error(['getImageSet : setType should be other train or test, but got ' setType]);
end

if (nargin < 5)
    pos_class = conf.class_subset;
else
    pos_class = classes;
end

train_images_path = fullfile(conf.imageSplitDir,[setType,'.txt']);
fid = fopen(train_images_path);
train_images = textscan(fid,'%s');
fclose(fid);
train_images = train_images{1};

labels = false(size(train_images));
for k = 1:length(pos_class)
    curClass = conf.classes{pos_class(k)};
    labels = labels | strncmp(curClass,train_images,length(curClass));
end
all_labels = zeros(size(labels));
for k = 1:length(conf.classes)
    all_labels(strncmp(conf.classes{k},train_images,length(conf.classes{k}))) = k;
end
ids = train_images;
ids = ids((1+offset):ratio:end);
labels = labels((1+offset):ratio:end);
end