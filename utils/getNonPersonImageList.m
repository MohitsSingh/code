function paths = getNonPersonImageList()
imgBaseDir = '/home/amirro/storage/data/VOCdevkit/VOC2012/JPEGImages';
[ids,t] = textread('/home/amirro/storage/data/VOCdevkit/VOC2012/ImageSets/Main/person_train.txt',...
    '%s %d');

ids = ids(t==-1);

paths = cellfun2(@(x) [x '.jpg'],fullfile(imgBaseDir,ids));
