
load('data/front.mat'); 
load('data/gt.mat');


local = false;

if (local)
    dataDir = 'D:\pose_data\data';
    imgDir = 'D:\h3d\images';
else    
    dataDir = 'data';
    imgDir = 'X:\data\poselets\data\person\image_sets\h3d\images';
end

masksDir = 'masks';

part_to_detect = 'R_Shoulder';

crop_size = [64 48];

%%
% Mark locations of positive examples. While at it, create
% for each image a mask containing the poeple in it (used for learning)

% create negative samples from the first n images...

fid_neg = fopen('neg_example.txt','w+');
fid_pos = fopen('pos_example.txt','w+');

nNegativeImages = 1000;
c = 0;
for k = 1:length(front)
    if (front(k))
        c
                       
        matPath = fullfile(dataDir,sprintf('train_person_%d.mat',k));
        load(matPath);
        
        img_name = training_data.img_name;
        imgPath = fullfile(imgDir,[img_name '.jpg']);
        if (c < nNegativeImages)
            fprintf(fid_neg, '%s\n', imgPath);
            % also print the name of the mask file
            fprintf(fid_neg, '%s\n', fullfile('X:\code\pose','masks',[img_name '.png']));
        end
                                
        c = c+1;
        
          % check if the current part file already exists...
%         if (exist(fullfile('pos_shoulders',sprintf('%05.0f.png',k)),...
%                 'file'))
%             continue;
%         end
                        
        faceMask = training_data.face_mask;
        rprops = regionprops(faceMask,'BoundingBox');
        % size of face
        bbsize = rprops.BoundingBox(3:4);
               
        [tf,part_idx]=ismember(part_to_detect,training_data.keypoints_labels);
        if (tf)
            part_loc= training_data.keypoints_gt(:,part_idx);
        end
        
                              
        
%         figure,imshow(imgPath);
%         hold on;
%         plot(part_loc(1),part_loc(2),'g*');
        I = imread(imgPath);
               
        % extract image of desired part around part location
        
        %TODO: careful not to crop outside the image. I hope this 
        % doesn't happen;  currently it's NOT handled. 
        
        %TODO - make the cropping size a per-part parameter
        
        crop_size_ = norm(bbsize)*crop_size/sqrt(800);
        
        [patch_ rect] = imcrop(I,[part_loc(1)-crop_size_(2)/2,...
            part_loc(2)-crop_size_(1)/2,...
            fliplr(crop_size_)]);
        
        fprintf(fid_pos, '%s %d %d %d %d %03.3f\n', imgPath,...
            rect(1),rect(2),rect(1)+rect(3),rect(2)+rect(4),...
            norm(bbsize)/sqrt(800));
        
%         imwrite(imresize(patch_,2*crop_size,'bicubic'),...
%             fullfile('pos_shoulders',sprintf('%05.0f.png',k)));
%         
        
% % %         maskFile = fullfile(masksDir, [img_name '.png']);
% % %         person_mask = training_data.body_mask;
% % %         if (exist(maskFile,'file'))
% % %             person_mask = max(person_mask , imread(maskFile));
% % %         end
% % %         imwrite(person_mask,maskFile);
    end
end

fclose(fid_neg);
fclose(fid_pos);

%%

d = dir('pos_shoulders\*.png');
fid = fopen('pos_example.txt','w+');
for k = 1:length(d)
    fprintf(fid,'%s\n',fullfile(pwd,'pos_shoulders',d(k).name));
end

fclose(fid);

% extract hogs ?

%% read positive, negative examples
% pos_samples = dlmread('pos_out.txt');
% neg_samples = dlmread('neg_out.txt');

all_samples = dlmread('D:\mat.txt');

pos_samples = all_samples(1:381,:);
neg_samples = all_samples(382:end,:);

train_pos_inds = 1:size(pos_samples,1)/2;
train_neg_inds = 1:size(neg_samples,1)/2;

train_pos_neg = [pos_samples(train_pos_inds,:);neg_samples(train_neg_inds,:)];
train_labels = [ones(length(train_pos_inds),1);zeros(length(train_neg_inds),1)];
parm_string = '-s 0 -t 0';
model = svmtrain(train_labels, train_pos_neg, parm_string);

parm_string = '-s 0 -t 0 -n 0.021696';
model = svmtrain(train_labels, train_pos_neg, parm_string);

%%

test_pos_inds = setdiff(1:size(pos_samples,1),train_pos_inds);
% test_pos_inds = (size(pos_samples,1)/2+1):(size(pos_samples,1)/2+5);
test_neg_inds = setdiff(1:size(neg_samples,1),train_neg_inds);
%%
close all;
figure;
hold on;
for k = 1:10
    inds_ = randperm(length(test_neg_inds));
    test_neg_inds_ = test_neg_inds(inds_(1:length(test_pos_inds)));
    
    test_pos_neg = [pos_samples(test_pos_inds,:);neg_samples(test_neg_inds_,:)];
    test_labels = [ones(length(test_pos_inds),1);zeros(length(test_neg_inds_),1)];
    
    [predicted_label, accuracy, decision_values] = ...
        svmpredict(test_labels, test_pos_neg,model,'-b 0');
    
    [tpr,fpr,thresholds] = roc([test_labels]' ,decision_values');
    
    plot(fpr,tpr);
end
%%
modelSV = full(model.SVs);
svWeightedSum = transpose(model.sv_coef)*modelSV;

%dlmwrite('svmparams.txt',[length(svWeightedSum); svWeightedSum';model.rho]);
dlmwrite('svmparams.txt',[svWeightedSum';-model.rho]);

D = squareform(pdist(pos_samples(train_pos_inds,:)));
figure,hist(D(:))

D = squareform(pdist(neg_samples(train_neg_inds,:)));
figure,hist(D(:))

