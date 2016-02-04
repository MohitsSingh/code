function allLabels = getAllLabels(conf,setType)
% return the labels of all images.


if (~strcmp(setType,'train') && ~strcmp(setType,'test'))
    error(['getImageSet : setType should be other train or test, but got ' setType]);
end
allLabels = [];
for pos_class = 1:length(conf.classes)
    train_images_path = fullfile(conf.imageSplitDir,[setType,'.txt']);
    fid = fopen(train_images_path);
    train_images = textscan(fid,'%s');
    fclose(fid);
    train_images = train_images{1};
    p = conf.classes{pos_class};
    labels = strncmp(p,train_images,length(p));
    allLabels(find(labels)) = pos_class;
end


end