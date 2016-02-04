function [ids,labels] = getImageSet(conf,setType)
%GETTRAINIMAGES Summary of this function goes here
%   Detailed explanation goes here
    pos_class = conf.classes{conf.class_subset};
    
    train_images_path = fullfile(conf.imageSplitDir,'train.txt');
    fid = fopen(train_images_path);
    train_images = textscan(fid,'%s');
    fclose(fid);
    train_images = train_images{1};
        
    labels = strncmp(pos_class,train_images,length(pos_class));
    ids = train_images; 
end