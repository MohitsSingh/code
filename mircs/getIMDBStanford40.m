imdb = struct;
imdb.dir = '/home/amirro/storage/data/Stanford40/JPEGImages';
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

all_ids = [train_ids;test_ids];
all_classes = [all_train_labels;all_test_labels];
imdb.sets.TRAIN = 1;
imdb.sets.TEST = 2;
imdb.images.id = 1:length(all_ids);
imdb.images.name = all_ids;
imdb.images.set = imdb.sets.TEST*ones(1,length(all_ids));
imdb.images.set(1:length(train_ids)) = imdb.sets.TRAIN;
imdb.images.size = zeros(2,length(all_ids));
for k = 1:length(all_ids)  
    k
    m = imfinfo(fullfile(imdb.dir,imdb.images.name{k}));
    imdb.images.size(1,k) = m.Width;
    imdb.images.size(2,k) = m.Height;
end
imdb.classes = struct;
imdb.classes.name = A;

for k = 1:length(imdb.classes.name)
    imdb.classes.imageIds{k} = find(all_classes==k);
end
