fid = fopen(sprintf(VOCopts.imgsetpath,VOCopts.trainset));
train_images = textscan(fid,'%s');
train_images = train_images{1};
fclose(fid);

fid = fopen(sprintf(VOCopts.imgsetpath,VOCopts.testset));
test_images = textscan(fid,'%s');
test_images = test_images{1};
fclose(fid);

%     train_images = textread(sprintf(VOCopts.imgsetpath,VOCopts.trainset),'%s');
%     test_images = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s');

all_images = [train_images;test_images];

%
% [bboxes,bboxesXclass]=get_bbox_data(globalOpts,all_images);
%     
%
%
%     bboxes = single(bboxes);
%     bboxesXclass = single(bboxesXclass);
%     sa
% train a vocabulary...
vocab = train_vocabulary(globalOpts,train_images(1:end));
kdtree = vl_kdtreebuild(vocab);
%
% quantize_descs(globalOpts,train_images,vocab,kdtree);
% quantize_descs(globalOpts,test_images,vocab,kdtree);
%

% get structures for specific training examples.
model.vocab = vocab;
model.kdtree = kdtree;

imageSelTrain = 1:length(train_images);


% imageSelTest = setdiff(1:length(train_images),imageSelTrain);

% save trainingInstances_200_rescale_0_overlap trainingInstances -v7.3;

% train for each class independently....