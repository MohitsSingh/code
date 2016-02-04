% Experiment 0047
% 20/7/2014
%Extract candidates regions to narrow down the regions for inspection.

if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    load fra_db.mat;
    all_class_names = {fra_db.class};
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    % make sure class names corrsepond to labels....
    [lia,lib] = ismember(classes,class_labels);
    classNames = all_class_names(lib);
    isTrain = [fra_db.isTrain];
    initialized = true;
    roiParams.infScale = 3.5;
    roiParams.absScale = 200*roiParams.infScale/2.5;
    normalizations = {'none','Normalized','SquareRoot','Improved'};
end


% [~,~,I] = get_rois_fra(conf,fra_db(100),roiParams);
% figure,imagesc2(I);

% addpath(genpath('/home/amirro/code/3rdparty/SelectiveSearchCodeIJCV/'));

%%
load classInfo.mat
debugging = true;
objTypes = {'head','hand','obj'};
% objTypes = {'obj'};
if (~debugging)
    obj_feats = cell(length(objTypes),length(fra_db),length(classes));
    gt_obj_feats = cell(length(fra_db),length(classes));
end
ticId = ticStatus('automatic object feature extraction',1,.1);
pk = 1:length(fra_db);
shortNames = {'drink','smoke','blow','brush','phone'};
mm = 1;
nn = 3;

for ik = 1:length(fra_db)
    k = pk(ik)
    if (fra_db(k).isTrain),continue,end;
    %             if (fra_db(k).classID~=cur),continue,end;
    curImageData = fra_db(k);
    
    [rois,subRect,I,scaleFactor] = get_rois_fra(conf,curImageData,roiParams);
    
    % get regions / bounding boxes for this image.
    sssPath = j2m('~/storage/s40_fra_selective_search',curImageData);
    
    % %get the new gpb data....
    %
    
    %     sssRes = load(sssPath);sssRes = sssRes.res;
    %     boxes = BoxRemoveDuplicates(sssRes.boxes);
    %     regions = cellfun2(@(x) myBlob2Image(x,I),sssRes.blobs);
    %     clf; imagesc2(I); drawnow; pause(.1);cont
    curFeatsPath = j2m('~/storage/s40_fra_pred_feats',curImageData.imageID);
    mcgPath = j2m('/home/amirro/storage/s40_seg_new',curImageData.imageID);
    load(mcgPath,'res');
    boxes = res.cadidates.bboxes(:,[2 1 4 3]);
    [boxes,uniqueIDX] = BoxRemoveDuplicates(boxes);
    boxScores = res.cadidates.scores(uniqueIDX);
    [~,~,orig_areas] = BoxSize(boxes);
    intersection = BoxIntersection(boxes, subRect);
    [~,~,areas] = BoxSize(intersection);
    %     I = getImage(conf,curImageData);
    rel_areas = areas./orig_areas;
    boxes = boxes(rel_areas>=.9,:);
    boxScores = boxScores(rel_areas>=.9);
    boxes = clip_to_image(boxes,subRect);
%     size(I)
%     max(boxes(:,3))
    boxes =  bsxfun(@minus,boxes,subRect([1 2 1 2]));
    boxes = round(boxes*scaleFactor);
    %     boxes = clip_to_image(boxes,subRect);
    % %
    %     [r,ir] = sort(boxScores,'descend');
    %     for t=  1:size(boxes,1);
    %         clf; imagesc2(I); hold on; plotBoxes(boxes(ir(t),:));
    %         pause
    %        drawnow
    %     end
    
    subs = {};
    vals = {};
    
    realClassID = curImageData.classID;
    overall_scores = zeros(size(classes));
    
    unaryScores = {};
    binaryScores = {};
    %     total_score = zeros(size(classes));
    for iClass = 1:length(classes)
        
        if (curImageData.classID ~= iClass)
            continue;
        end
        
        vals = {};
        for iObjType = 3
            probPath = fullfile('~/storage/s40_fra_box_pred_new',[curImageData.imageID '_' classNames{iClass} '_' objTypes{iObjType} '.mat']);
            L = load(probPath);
            pMap = imResample(L.pMap,size2(I));
            % Perform Selective Search
            sums = sum_boxes(pMap,boxes);
            [w h a] = BoxSize(boxes);
            boxes_1 = [boxes 0*(sums./a)+boxScores*10];
            H = computeHeatMap(I,boxes_1,'sum');
            figure,imagesc2(H);figure,imagesc2(I);
            
            
            %             bb = [0 0 1 1];
            %             [w h a] = BoxSize(bb);
            
            maxSup = 3;
            [curSubs,curVals] = nonMaxSupr( double(pMap), maxSup,[],nMax);
            if (length(curVals)) < nMax
                z = -inf(nMax,1); z(1:length(curVals)) = curVals;
                curVals = z;
            end
            vals{iObjType} = curVals;
            
            regions = cellfun2(@(x) myBlob2Image(x,I),hBlobs);
            region_sums = cellfun(@(x) sum(pMap(x(:))),regions);
            region_sizes = cellfun(@(x) sum(x(:)),regions);
            displayRegions(I,regions,region_sums./region_sizes)
            
            displayRegions(I,regions)
            myBlob2Image(hBlobs{10});
            
            
            if (size(curSubs,1)) < nMax
                z = -inf(nMax,2); z(1:size(curSubs,1),:) = curSubs;
                curSubs = z;
            end
            subs{iObjType} =  fliplr(curSubs );
            all_unary_scores(k,iClass,iObjType) = max(pMap(:));
            overall_scores(iClass) = overall_scores(iClass)+max(pMap(:));
            
            %             subplot(1,3,iObjType);
            %             imagesc2(sc(cat(3,pMap,I),'prob'));title(objTypes{iObjType});
            %             hold on;plotPolygons(fliplr(subs{iObjType}),'g*');
        end
        
        vals = cat(2,vals{:});
        unaryScores{iClass} = vals(a,1)+vals(b,2)+vals(c,3);
        d1 = subs{2}-subs{1};
        d2 = subs{3}-subs{1};
        d3 = subs{3}-subs{2};
        d1 = subs{2}(b,:)-subs{1}(a,:);
        d2 = subs{3}(c,:)-subs{1}(a,:);
        d3 = subs{3}(c,:)-subs{1}(b,:);
        d = [d1 d2 d3];
        binaryScores{iClass} = d*all_w(:,iClass);
        total_score(k,iClass) = max(binaryScores{iClass}+unaryScores{iClass});
    end
    
    [m,im] = max(total_score);
    %     clf; plot(total_score);
    %     hold on; plot(im,m,'*');pause
    
    
    continue;
    
    %     curBinaryScores = d*
    
    all_scores(k,:) = overall_scores;
    [v,iv] = max(overall_scores);
    confMat(realClassID,iv) = confMat(realClassID,iv)+1;
    continue
    
    overall_scores = normalise(overall_scores);
    
    clf; plot(overall_scores);drawnow;continue;
    
    
    R = load(curFeatsPath);
    for iClass = 1:length(classes)
        for iObjType = 3
            obj_feats{iObjType,k,iClass} = R.res(iClass).feats;
        end
    end
    
    set(gcf,'name',curImageData.class);
    C = linspecer(5);
    %     colormap(C);
    figure(1);clf;
    %subplot(mm,nn,1);
    imagesc2(I);
    plotBoxes(rois(1).bbox,'y');
    class_scores = zeros(size(classes));
    for iClass = 1:length(classes)
        L = load(probPath);
        curBoxes = R.res(iClass).bbox(5,:);
        %curFeats = R.res(iClass).feats(:,[5 14]);
        curFeats = R.res(iClass).feats;
        curScore = max(ws(iClass,:)*curFeats);
        plotBoxes(curBoxes,'LineStyle','--','LineWidth',2,'Color',C(iClass,:));
        text(curBoxes(1)+1,curBoxes(2)+10,shortNames(iClass),'Color',C(iClass,:),'FontWeight','bold','FontSize',15);
        %         text(curBoxes(1)+1,curBoxes(4)-10,num2str(curScore),'Color',C(iClass,:),'FontWeight','bold','FontSize',15);
        class_scores(iClass) = curScore;
    end
    
    
    
    figure(5);clf;
    
    
    
    %     p%ause
    vals = cat(2,vals{:}); % not so many options, iterate over local maxima
    unaryScores = vals(a,1)+vals(b,2)+vals(c,3);
    
    
    %         locs1 = subs
    
    % now find the configuration...
    
    %     pause;
    %     continue;
    
    f_obj = strmatch('obj',{rois.name});
    cur_gt_scores = gt_scores(:,k);
    if (~isempty(f_obj))
        plotBoxes(cat(1,rois(f_obj).bbox) ,'LineStyle','-','LineWidth',2,'Color','k')
    end
    figure(2); clf;
    %subplot(mm,nn,2);
    xlim([-1 length(classes)+1]);
    ylim([-2 2]);axis equal;
    set(gca,'XTick',[1:5]);
    set(gca,'XTickLabel',shortNames);
    %
    
    hold on;
    %     for iClass = 1:length(classes)
    %         stem(iClass,class_scores(iClass),'Color',C(iClass,:));
    %     end
    stem(1:length(classes),class_scores,'b-');
    plot(1:5,cur_gt_scores,'r-','LineWidth',2);
    
    plot(realClassID,cur_gt_scores(realClassID),'m*');
    
    
    rotateticklabel(gca,90);
    
    fSal = foregroundSaliency(conf,curImageData.imageID);
    figure(3); clf; imagesc2(cropper(fSal,subRect));
    
    figure(4);
    clf
    maxMaps = zeros(size(classes));
    s = zeros(size2(I));
    for iClass = 1:length(classes)
        probPath = fullfile('~/storage/s40_fra_box_pred_new',[curImageData.imageID '_' classNames{iClass} '_' 'obj' '.mat']);
        L = load(probPath);
        subplot(2,3,iClass);
        maxMaps(iClass) = max(L.pMap(:));
        pMap = imResample(L.pMap,size2(I));
        s = s+pMap;
        imagesc2(sc(cat(3,pMap,I),'prob')); colorbar;title(shortNames{iClass});
    end
    
    maxMaps = normalise(maxMaps);
    %     subplot(2,3,6); imagesc2(sc(cat(3,s,I),'prob'));
    %     figure(1);
    %     subplot(1,2,2);
    figure(2);plot(maxMaps,'g-o');
    plot(maxMaps+cur_gt_scores','m-');
    plot(overall_scores,'k--','LineWidth',2);
    legend({'scores','gt scores','GT','maps','gt scores+maps'},'overall');
    
    
    
    pause
    % get the "real" scores for this class.
    
    
    
    %     if (debugging)
    %         clf; imagesc2(I);
    %         realClass = curImageData.class;
    %     end
    
end
[ overlaps ,ints] = boxesOverlap( bboxes,bboxes2)%%
iClass = 1;
% figure,plot(gt_scores(1,:));
figure,plot((gt_scores(iClass,:)>=max(gt_scores(setdiff(1:length(classes),iClass),:),[],1))*.5);

[m,im] = max(gt_scores);

figure,plot(im==1)

hold on; plot([fra_db.classID]==1,'g');

