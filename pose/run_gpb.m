function run_gpb(k)

addpath(genpath('/home/amirro/code/3rdparty/grouping'));
load('data/front.mat');
load('data/gt.mat');

dataDir = 'data';
imgDir = '/home/amirro/data/poselets/data/person/image_sets/h3d/images';
gpbDir = '/home/amirro/data/poselets/data/person/image_sets/h3d/gpb';

if(~exist(gpbDir,'dir'))
    mkdir(gpbDir);
end

part_to_detect = 'Nose';
debug_ = false;
crop_size = [200 200];

for k = 1:length(front)
    k
    if (front(k))
        matPath = fullfile(dataDir,sprintf('train_person_%d.mat',k));
        load(matPath);
        img_name = training_data.img_name;
        
        imgPath = fullfile(imgDir,[img_name '.jpg']);
        I = imread(imgPath);
        
        gpbPath = fullfile(gpbDir,[img_name num2str(k,'%05.0f') '.mat']);
        gpbImagePath = fullfile(gpbDir,[img_name num2str(k,'%05.0f') '.jpg']);
        
        faceMask = training_data.face_mask;
        rprops = regionprops(faceMask,'BoundingBox');
        % size of face
        bbsize = rprops.BoundingBox(3:4);
        
        
        [tf,part_idx]=ismember(part_to_detect,training_data.keypoints_labels);
        if (tf)
            part_loc= training_data.keypoints_gt(:,part_idx);
        else
            return;
        end
        
        resize_factor = norm(bbsize)/sqrt(800);
        crop_size_ = resize_factor*crop_size;
        
        [patch_ rect] = imcrop(I,[part_loc(1)-crop_size_(2)/2,...
            part_loc(2)-crop_size_(1)/2,...
            fliplr(crop_size_)]);
        
        %imgPath = strrep(gpbPath,'.mat','.png');
        if (~exist(gpbPath,'file'))
            imwrite(imresize(patch_,1/resize_factor, 'bilinear'),gpbImagePath);
            [gPb_orient, gPb_thin, textons] = globalPb(gpbImagePath);
            save(gpbPath, 'gPb_orient', 'gPb_thin', 'textons','rect');
        else
            if (debug_)
                close all;
                I = imread(gpbImagePath);
                figure,imshow(I);
                load(gpbPath);
                figure,imshow(gPb_thin>0,[]);
                pause;
            end
        end
    end
end
%     end

% end
