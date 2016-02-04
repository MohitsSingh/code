initpath;
config;
conf.suffix = 'train_4';
% dataset of images with ground-truth annotations
% start with a one-class case.
% start from first image and get every second image
[train_ids,train_labels] = getImageSet(conf,'train',2,0);
% for testing, start from second image and get every second image
[val_ids,val_labels] = getImageSet(conf,'train',2,1);
