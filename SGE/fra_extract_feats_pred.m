function res = fra_extract_feats_pred(conf,I,reqInfo)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    res.conf = conf;
    load fra_db.mat;
    res.fra_db = fra_db;
    conf.get_full_image = true;
    all_class_names = {fra_db.class};
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    % make sure class names corrsepond to labels....
    [lia,lib] = ismember(classes,class_labels);
    classNames = all_class_names(lib);
    res.classNames = classNames;
    return;
end
fra_db = reqInfo.fra_db;
[learnParams,conf] = getDefaultLearningParams(conf,1024);
% make sure class names corrsepond to labels....
roiParams.infScale = 3.5;
roiParams.absScale = 200*roiParams.infScale/2.5;
featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
roiParams.useCenterSquare = false;
debugging = false;
objTypes = {'head','hand','obj'};
classNames = reqInfo.classNames;
res = [];
if (curImageData.isTrain),return,end;
[rois,subRect,I,scaleFactor] = get_rois_fra(conf,curImageData,roiParams); %TODO: make sure this window is the same

for iClass = 1:length(classNames)
    if (debugging)
        if (curImageData.classID~=iClass),continue,end;
    end
    for iObjType = 3:length(objTypes)
        probPath = fullfile('~/storage/s40_fra_box_pred_new',[curImageData.imageID '_' classNames{iClass} '_' objTypes{iObjType} '.mat']);
        L = load(probPath);
        L.boxes_orig(:,1:4) = inflatebbox(L.boxes_orig,1.5,'both',false);
        L.boxes_orig = L.boxes_orig(1,:);
        [h  w]= BoxSize(L.boxes_orig);
        h = h(1);
        if (debugging)
            I_orig = getImage(conf,curImageData);
            figure(1);clf; imagesc2(I_orig);
            
            if (~isempty(curImageData.objects))
                polys = curImageData.objects(1).poly;
                gt_objBox = pts2Box(polys);
                plotBoxes(gt_objBox,'g-','LineWidth',2);
                gt_objBox_orig = gt_objBox;
                gt_objBox = inflatebbox(gt_objBox,[h h],'both',true);
                plotBoxes(gt_objBox,'c--','LineWidth',2);
                gt_objBox = bsxfun(@minus,gt_objBox,[subRect([1 2 1 2])])*scaleFactor;
                gt_objBox_orig = bsxfun(@minus,gt_objBox_orig,[subRect([1 2 1 2])])*scaleFactor;
            end
            
            II = cropper(I_orig,L.roiBox);
            pMap = imResample(L.pMap,size2(I));
            figure(2);clf; imagesc2(sc(cat(3,pMap,I),'prob'));%pause;
        end
        bb = bsxfun(@minus,L.boxes_orig,[subRect([1 2 1 2]) 0])*scaleFactor;
        max_d = 10;
        delta_d = 10;
        [dx,dy] = meshgrid(-max_d:delta_d:max_d,-max_d:delta_d:max_d);
        dx = 0; dy = 0;
        bb_orig = bb(1:4);
        bb_new = repmat(bb(:,1:4),numel(dx),1);
        bb_new = bb_new+[dx(:) dy(:) dx(:) dy(:)];
        
        %                 figure,imagesc2(I);
        %                 plotBoxes(bb_new);
        %                 bb_new =bb;
        rois = {};
        roiSummary = zeros(size2(I));
        for iBox = 1:size(bb_new,1)
            rois{iBox} = poly2mask2(box2Pts(bb_new(iBox,1:4)),size2(I));
            roiSummary = roiSummary+iBox*rois{iBox};
        end
        %         rois = {};
        %         roiSummary = zeros(size2(I));
        %         for iBox = 1:size(bb,1)
        %             rois{iBox} = poly2mask2(box2Pts(bb(iBox,1:4)),size2(I));
        %             roiSummary = roiSummary+iBox*rois{iBox};
        %         end
        
        if (debugging)
            figure(4); clf; imagesc2(I);
            hold on;
            plotBoxes(gt_objBox,'y--','LineWidth',2);
            plotBoxes(gt_objBox_orig,'g--','LineWidth',2);
            plotBoxes(bb(1,:),'r--','LineWidth',1);
            legend({'inf','orig','predicted'});                pause; continue;
            %                 continue;
        end
        
        curFeats = featureExtractor.extractFeatures(I,rois);
        % flip! (although rearranging the loop order will make this more efficient, but not important right now)
        curFeats = [curFeats,featureExtractor.extractFeatures(flip_image(I),cellfun2(@flip_image,rois))];
        
        % for sanity, get the corresponding bounding box, had the region been detected centered as in the manual
        % setting
        %         if (~isempty(curImageData.objects))
        %             polys = curImageData.objects(1).poly;
        %             gt_objBox = pts2Box(polys);
        %             gt_objBox = inflatebbox(gt_objBox,[h h],'both',true);
        %             gt_objBox = bsxfun(@minus,gt_objBox,[subRect([1 2 1 2])])*scaleFactor;
        %             gt_objBox = inflatebbox(gt_objBox,[h h],'both',true);
        %             gt_obj_feats{k,iClass} = featureExtractor.extractFeatures(I,poly2mask2(box2Pts(gt_objBox),size2(I)));
        %         end%
        
        %         if (~isempty(curImageData.objects))
        %             for iObj = 1:length(curImageData.objects)
        %                 gt_objBox = pts2Box(polys);
        %         end
        
    end
    res(iClass).bbox = [bb_new;flip_box(bb_new,size2(I))];
    res(iClass).class = iClass;
    res(iClass).feats = curFeats;
end
end% save ~/storage/misc/fra_db_w

