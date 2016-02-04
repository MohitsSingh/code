if (~exist('initialized','var'))
    cd ~/code/mircs
    initpath;
    config;
    
    %     load ~/code/mircs/images_and_face_obj_full_imdb.mat
    %     load ~/storage/misc/images_and_masks_x2_w_hands.mat % specifically, fra_db was loaded
    if (0)
        load ~/code/mircs/images_and_face_obj_full.mat
        train = find(isTrain);
        val = train(1:3:end);
        test = find(~isTrain);
        train = setdiff(train,val);
        imdb.images = 1:length(images);
        imdb.images_data = images;
        needToConvert = isa(images{1},'double') && max(images{1}(:))<=1;
        if needToConvert
            for t = 1:length(images)
                imdb.images_data{t} = im2uint8(imdb.images_data{t});
            end
        end
        for t = 1:length(masks)
            m = masks{t};
            z = m == 0;
            obj = m == 1;
            m = m-1;
            m(z) = 0;
            m(obj) = 2+fra_db(t).classID;
            masks{t} = m;
            %               clf; subplot(1,2,1); imagesc2(masks{t}); colorbar;
            %               subplot(1,2,2); imagesc2(m); colorbar;
            %               dpc
        end        
        masks = cellfun2(@uint8,masks);
        imdb.labels = masks;
        imdb.nClasses = 7;
        save('~/storage/misc/images_and_face_obj_full_imdb.mat','imdb','train','val','test','fra_db');
    end
    load('~/storage/misc/images_and_face_obj_full_imdb.mat')
    
    seg_dir = '~/storage/fra_db_seg_full';
    isTrain = find([fra_db.isTrain]);

    initialized = true;
end

%
% load the segmentations...

%%
data_stats = struct('imgIndex',{},'groundTruth',{},'goodRegions',{},'goodOverlaps',{},'nRegions',{});
%%
for iTrain = 1:length(isTrain)
    iTrain
    k = isTrain(iTrain);
    imgData = fra_db(k);
    I = getImage(conf,imgData);
    % load image
%     clf; imagesc2(I);
    % get the MCG segmentation
    %%%    
    curGroundTruth = imdb.labels{k};
    curObj = curGroundTruth>=3;    
    load(j2m(seg_dir,imgData)); % candidates, ucm2
    candidates = segs.candidates;
    ucm2 = segs.ucm2;
    z = {};
    for u = 1:size(candidates.masks,3)
        z{end+1} = candidates.masks(:,:,u);
    end                
    if none(curObj)
        ovps = zeros(size(z));
    else
        [ovps,ints,uns] = regionsOverlap(curObj,z);    
    end
    data_stats(k).imgIndex = k;
    data_stats(k).groundTruth = curObj;    
    data_stats(k).goodOverlaps = ovps;
    data_stats(k).nRegions = length(z);
%     displayRegions(I,z,ovp,'maxRegions',3);
end
%save ~/storage/misc/data_stats_full_image.mat data_stats 
save ~/storage/misc/data_stats_full_image_2.mat data_stats 

%%
xo = linspace(0,1,20);
%%
for iClass = 1:5
    id_to_class_name{iClass} = fra_db(find([fra_db.classID]==iClass,1,'first')).class;    
end

data_train = data_stats(isTrain);
fra_db_train = fra_db(isTrain);
max_ovps = cell(5,1);
%max_ovps_for_class = zeros(size(data_train));
for t = 1:length(data_train)
    ovps = data_train(t).goodOverlaps;
    m =0;
    if any(ovps)
        m=max(ovps);
    end
    max_ovps{fra_db_train(t).classID}{end+1} = m;
end

max_ovps = cell2mat(cat(1,max_ovps{:}));
figure,[no,xo]=hist(max_ovps',20);
bar(xo,bsxfun(@rdivide,no,sum(no)))
legend(id_to_class_name,'Interpreter','none');


%%
%%%
%[candidates,ucm2,isvalid] = getCandidateRegions(conf,imgData,I_sub,isTrain)