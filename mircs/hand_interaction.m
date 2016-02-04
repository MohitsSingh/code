% 6/5/2015
%---------
% Action Object Detection - a structured output framework for action object detection
% and description.
% % % initpath;
% % % config;
% % % addpath('/home/amirro/code/3rdparty/edgeBoxes/');
%%
% % % load('~/data/ita/train_data.mat');
% % % load('~/data/ita/test_data.mat');
%%
toVisualize = true;
regionSampler.boxOverlap = .5;
regionSampler.boxSize = 28*[1 1];
boxSize = regionSampler.boxSize;
regionSampler.minRoiOverlap = .2;
pos_patches = {};
neg_patches = {};
p_ratio = 2;
for iTrain = 1:length(train_data)
    iTrain
    curData = train_data{iTrain};
    I = curData.image;
    z = curData.seg_labels ~= 0;
    hand_mask = curData.seg_labels < 7 & z;
    hand_mask = imclose(hand_mask,ones(3));
    obj_mask = curData.seg_labels ==7 & z & ~hand_mask;
    obj_mask = imopen(obj_mask,ones(2));        
%     obj_mask = imopen(obj_mask,ones(3));
    b_obj = bwdist(obj_mask);
    mm_pos = bwdist(hand_mask) <=1 & b_obj <= 1;
    [yy,xx] = find(mm_pos);
    
    bb_pos = inflatebbox([xx yy],regionSampler.boxSize,'both',true);
    sums = sum_boxes(double(obj_mask),bb_pos);
    %ovps = boxRegionOverlap(bb_pos,obj_mask);
    %bb_pos = bb_pos(nms([bb_pos ones(size(bb_pos,1),1)],.5),:);
    pos_keep = nms([bb_pos sums],.7);
    pos_keep = pos_keep(1:min(3,length(pos_keep)));
    bb_pos = bb_pos(pos_keep,:);
    %mm_neg = bwdist(hand_mask) <=1 & bwdist(obj_mask) > regionSampler.boxSize(1)/2;
    mm_neg = bwperim(hand_mask) & bwdist(obj_mask) > regionSampler.boxSize(1)/2;
    %mm_neg = bwperim(mm_neg);
    
%     z_select = false(size2(I));
%     jmp = round(boxSize(1)/p_ratio);
%     z_select(1:jmp:end,1:jmp:end) = true;
    [yy,xx] = find(mm_neg);
    bb_neg = inflatebbox([xx yy],regionSampler.boxSize,'both',true);
    bb_neg = bb_neg(nms([bb_neg ones(size(bb_neg,1),1)],.5),:);
    
    [ovps,ints] = boxesOverlap(bb_pos,bb_neg);
%     keep_pos = max(ovps,[],2) < .15;
    keep_neg  = max(ovps,[],1) < .15;
    
%     bb_pos = bb_pos(keep_pos,:);
    bb_neg = bb_neg(keep_neg,:);
    
    %obj_mask = imfilter(double(obj_mask),ones(15)/225);
% % %     v = 15;
% % %     roi_mask = imfilter(double(hand_mask),ones(v)/(v^2));
% % %     borders = region2Box(obj_mask);
% % % %     hand_box = region2Box(hand_mask);
% % %     
% % %     regionSampler.setRoiMask(roi_mask);            
% % %     r = regionSampler.sampleOnGrid(borders);    
% % %     box_labels = r(:,5);
% % %     r = round(r(:,1:4));
    
% % %     pos_patches{end+1} = multiCrop2(I,r(box_labels==1,:));
% % %     neg_patches{end+1} = multiCrop2(I,r(box_labels==0,:));
    pos_patches{end+1} = multiCrop2(I,bb_pos);
    neg_patches{end+1} = multiCrop2(I,bb_neg);
    
    if toVisualize        
        clf;
            subplot(1,2,1);
        %imagesc2(I);     
        displayRegions(I,obj_mask)
        %plotBoxes(r(bb==0,:));
        
        plotBoxes(bb_neg,'r-','LineWidth',2);
        plotBoxes(bb_pos,'g-','LineWidth',2);
        subplot(1,2,2);
        displayRegions(I,mm_neg);
        dpc
    end
end

pos_patches = cat(2,pos_patches{:});
neg_patches = cat(2,neg_patches{:});

%%
featureExtractor = DeepFeatureExtractor(conf);
pos_feats = featureExtractor.extractFeaturesMulti(pos_patches);
neg_feats = featureExtractor.extractFeaturesMulti(neg_patches);
[x,y] = featsToLabels(pos_feats,neg_feats);
[w b info] = vl_svmtrain(x, y, .01);

%%
regionSampler.boxOverlap = .5;
test_feats = struct('ind',{},'boxes',{},'feats',{});%'heatMap',{});
for iTest = 1:length(test_data)
    iTest
    curData = test_data{iTest};
    I = curData.image;
    z = curData.seg_labels ~= 0;
    hand_mask = curData.seg_labels < 7 & z;
    hand_mask = bwperim(imclose(hand_mask,ones(3)));
    [yy,xx] = find(hand_mask);   
    curBoxes = inflatebbox([xx yy],regionSampler.boxSize,'both',true);        
    curBoxes = curBoxes(nms([curBoxes ones(size(curBoxes,1),1)],.7),:);    
    patches = multiCrop2(I,curBoxes);       
    curFeats = featureExtractor.extractFeaturesMulti(patches);
    test_feats(iTest).ind = iTest;
    test_feats(iTest).boxes = curBoxes;
    test_feats(iTest).feats = curFeats;        
    fprintf('\n');       
end

% save ~/storage/misc/test_feats.mat test_feats
%%
toVisualize=false;
[w b info] = vl_svmtrain(x, y, .1);
%%
test_res = struct('ind',{},'boxes',{},'heatMap',{});
[w b info] = vl_svmtrain(x, y, .1);
for iTest = 1:length(test_data)
    iTest
    curData = test_data{iTest};
    I = curData.image;        
    curFeats = test_feats(iTest).feats;
    curScores = w'*curFeats;    
    boxesWithScores = [test_feats(iTest).boxes(:,1:4) curScores(:)];
    
    pick = nms(boxesWithScores,.8);
    curScores = normalise(curScores);
    z = computeHeatMap(I,boxesWithScores,'max');
    SS = sc(cat(3,z,im2double(I)),'prob');
    test_res(iTest).ind = iTest;
    test_res(iTest).boxes = boxesWithScores;
    test_res(iTest).heatMap = im2uint8(SS);
    if toVisualize
        clf;
        subplot(2,2,1);
        imagesc2(I);
        subplot(2,2,2); 
        imagesc2(SS);
        title(num2str(max(w'*curFeats)));
        subplot(2,2,3); imagesc2(I);
        plotBoxes(boxesWithScores(pick(2),:),'r-');
        plotBoxes(boxesWithScores(pick(3),:),'b-');
        plotBoxes(boxesWithScores(pick(1),:));
        dpc
    end        
end

% save test_res tedst_res
%% 

% x2({test_res(50:80).heatMap})

%%
for iTest = 1:length(test_data)
    iTest
    curData = test_data{iTest};
    I = curData.image;        
%     curFeats = test_feats(iTest).feats;
%     curScores = w'*curFeats;    
    %boxesWithScores = [test_feats(iTest).boxes(:,1:4) curScores(:)];
    boxesWithScores = test_res(iTest).boxes;    
    if toVisualize
        clf;
        subplot(2,2,1);
        imagesc2(I);
        subplot(2,2,2); 
        SS = test_res(iTest).heatMap
        imagesc2(SS);
        %title(num2str(max(w'*curFeats)));
        subplot(2,2,3); imagesc2(I);
        plotBoxes(boxesWithScores(pick(2),:),'r-');
        plotBoxes(boxesWithScores(pick(3),:),'b-');
        plotBoxes(boxesWithScores(pick(1),:));
        dpc
    end        
end