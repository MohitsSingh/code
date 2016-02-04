function my_data=my_get_full_data
conf.USE_BBOX=1;
conf.USE_OBJ=1;
conf.PARTIAL_DATA=1;
conf.BIG_SIZE=1;


load ~/code/mircs/fra_db.mat;

if conf.BIG_SIZE
    IMG_SZ=224+7*32; % resize to this size
    SEG_IMG_SZ=IMG_SZ;
    downsample=32; %from input image to top of conv5
    upsample_kernel=64;
    fc6_half_kernel=3;
    % fc6 out size: (IMG_SZ+2*pad)/downsample-2*fc6_half_kernel;
    % upsample out size: downsample*(fc6 out size)+(upsample_kernel)
    % downsample*((IMG_SZ+2*pad)/downsample-2*fc6_half_kernel)+(upsample_kernel)=SEG_IMG_SZ
    % IMG_SZ+2*pad-2*fc6_half_kernel*downsample+upsample_kernel=SEG_IMG_SZ
    % pad=SEG_IMG_SZ-IMG_SZ+2*fc6_half_kernel*downsample-upsample_kernel
    pad=(SEG_IMG_SZ-IMG_SZ+2*fc6_half_kernel*downsample-upsample_kernel)/2;
else
    IMG_SZ=150; % resize to this size
end




% retain only images which contain all these points
% conf.retain=[4 5 6 7];
conf.TYPES=1;

conf.IGNORE_VALUE=99;

if conf.BIG_SIZE
    load('data/train_data_x2_001.mat')
    load('data/train_data.mat','label')
else
    load('data/train_data.mat')
end
if conf.PARTIAL_DATA
    retain_pts = false(length(C.KeypointsToUse),1);  retain_pts(conf.retain)=true;
    train_data=train_data(ismember(label,conf.TYPES));
    train_data=retain_use_pts(train_data,retain_pts);
end
if conf.BIG_SIZE
    load('data/test_data_x2_001.mat')
    load('data/test_data.mat','label')
else
    load('data/test_data.mat')
end
if conf.PARTIAL_DATA
    test_data=test_data(ismember(label,conf.TYPES));
    test_data=retain_use_pts(test_data,retain_pts);
end
my_data=cat(1,train_data,test_data);
if conf.PARTIAL_DATA
    my_data([25 26 35 39 59 61 73 74 88 105 113 114 117 118 128 130])=[];
end
for j=1:length(my_data)
    image=my_data{j}.image;
    seg=my_data{j}.seg_labels;
    if conf.USE_BBOX
        bbox=my_data{j}.bbox;
        W_PCT=0.1; H_PCT=0.3;
        wh=[bbox(3)-bbox(1) bbox(4)-bbox(2)];
        bbox(1)=max(1,bbox(1)-wh(1)*W_PCT); bbox(3)=min(size(image,2),bbox(3)+wh(1)*W_PCT);
        bbox(2)=max(1,bbox(2)-wh(2)*H_PCT); bbox(4)=min(size(image,1),bbox(4)+wh(2)*H_PCT);
        bbox=floor(bbox);
        mask=true(size(seg));
        mask(bbox(2):bbox(4),bbox(1):bbox(3))=false;
        seg(mask & seg==7)=conf.IGNORE_VALUE;
        if 0
            figure(1); clf;imshow(image)
            figure(2); clf;imagesc(seg); axis image;
            pause
        end
    end
    imgsz=bwimgsz(image);
    %     ratio=IMG_SZ./max(imgsz);
    dest_hw=[IMG_SZ IMG_SZ];
    ratio=dest_hw./imgsz;
    image=imresize(image,dest_hw);
    seg=imresize(seg,dest_hw,'nearest');
    if ~conf.USE_OBJ
        seg(seg==7)=0;
    end
    my_data{j}.image=image;
    my_data{j}.seg_labels=seg;
    my_data{j}.pts.xy=bsxfun(@times,ratio,my_data{j}.pts.xy);
    if 0
        clf;imshow(image); hold on; plotxy(both_data{j}.pts.xy);
    end
end
end