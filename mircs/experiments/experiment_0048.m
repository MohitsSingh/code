
%% 29/7/2014
% tell apart action classes by close inspection of the relevant segments.

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
    roiParams.infScale = 3.5;
    roiParams.absScale = 200*roiParams.infScale/2.5;
    normalizations = {'none','Normalized','SquareRoot','Improved'};
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV/'));
    addpath('/home/amirro/code/3rdparty/logsample/');
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    featureExtractor = learnParams.featureExtractors{1};
    featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
    initialized = true;
    conf.get_full_image = true;
    load fra_db
    
end
%% initialization of regions around mouth area
% experiment parameters.

expResults = struct('params',{},'classifierResults',{});
% switch parameters and record performance...
iExp = 0;
resDir = '~/storage/s40_fra_box_pred_2014_09_17';
%mouthKnowns = [0 1];
mouthKnowns = 1;
%objectKnowns = [0 1];
objectKnowns = 0;
% squareMasks = [0 1];
squareMasks = 0;
%rotateObjs = [0 1];
rotateObjs = 0;

ppp = randperm(length(fra_db));
for mouthKnown = mouthKnowns
    for objectKnown = objectKnowns
        for squareMask = squareMasks
            for rotateObj = rotateObjs                                                                
                iExp = iExp+1;
                doStuff = true;
                % set all parameters
                                                
                expParams.mouthMode = mouthKnown;
                expParams.objectKnown = objectKnown;
                expParams.squareMask = squareMask;
%                                 
%                 expParams.mouthMode = false;
%                 expParams.objectKnown = false;
%                 expParams.squareMask = false;
                %
                if (doStuff)
                    % %
                    mouth_images = {};
                    all_images = {};
                    for it = 1:length(fra_db)
                        t = ppp(it)
%                         t = 856;
                        if (t <=500)
                            continue
                        end
%                         t = 502
                        if (mod(t,10)==0)                    
                            t
                        end
                        %if (~isempty(fra_db(t).feats)),continue,end
%                         if (fra_db(t).isTrain),continue,end;
                        if (fra_db(t).classID~=4 ),continue,end;
                        if (fra_db(t).classID==5),continue,end;
                        [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t));
                        rois_obj = rois([rois.id]==3);
                        fra_db(t).hasObj = false;
                        fra_db(t).isValid = false;
                        if isempty(rois_obj)
                            continue
                        end
                        fra_db(t).hasObj = true;
                        curPoly = {rois_obj.poly};
                        curBoxes = cellfun2(@pts2Box,curPoly);
                        curBoxes = cat(1,curBoxes{:});
                        [~,~,a_before] = BoxSize(curBoxes);
                        imbb = [1 1 size(I,2),size(I,1)];
                        curBoxes = BoxIntersection(curBoxes,imbb);
                        [~,~,a] = BoxSize(curBoxes);
                        goods = (a./a_before)>.5;
                        if (none(goods))
                            continue;
                        end
                        fra_db(t).isValid = true;
                        curPoly = curPoly(goods);
                        gt_mask = cellfun2(@(x)poly2mask2(x,size2(I)), curPoly);
                        gt_mask = max(cat(3,gt_mask{:}),[],3);
                        % get the mouth region
                        
                        if (expParams.mouthMode || fra_db(t).isTrain == true)
                            rois_mouth = rois([rois.id]==4);
                            mouth_bbox = inflatebbox(rois_mouth.bbox,1.5,'both',false);
                        else
                            resPath = fullfile('~/storage/s40_fra_box_pred_small_extent',[fra_db(t).imageID '_' classNames{1} '_' 'mouth' '.mat']);
                            L = load(resPath);
                            mouth_bbox = L.boxes_orig(1,1:4); % translate to coordinate system of roiBox                            
                            II = getImage(conf,fra_db(t));
                            %             figure(1);clf; imagesc2(II); hold on; plotBoxes(mouth_bbox);
                            %             plotBoxes(roiBox,'r-','LineWidth',2);
                            
                            mouthBoxCenter = (boxCenters(mouth_bbox)-roiBox(1:2))*scaleFactor;
                            bbox = [mouthBoxCenter mouthBoxCenter];
                            mouth_bbox = inflatebbox(bbox,[40 40],'both',true); % since it's originally a point.
                            mouth_bbox = round(inflatebbox(mouth_bbox,1.5,'both',false));
                        end
                        %     clf ;imagesc2(I); plotBoxes(mouth_bbox)
                                                
                        real_gt_mask = cropper(gt_mask,round(mouth_bbox));                        
                        if (expParams.objectKnown || fra_db(t).isTrain == true)
                            gt_mask = real_gt_mask;
                        else
                            resPath = fullfile(resDir,[fra_db(t).imageID '_' 'obj' '.mat']);
                            L_obj = load(resPath);
                            I_orig = getImage(conf,fra_db(t));
                            [objPredictionImage,maskPredictionImage] = predictionToImageSpace(L_obj,I_orig,roiBox);
                            f = @(x) (normalise(imResample(x,size2(I),'bilinear'))).^4;
                                                                                                                
                            objPredictionImage = f(objPredictionImage);
                            maskPredictionImage = f(maskPredictionImage);
                            
                            subCrop = round(inflatebbox(imbox(I),.6,'both',false));
                            I_orig = I;
                            I = cropper(I,subCrop);
                            maskPredictionImage_orig = maskPredictionImage;
                            maskPredictionImage = cropper(maskPredictionImage,subCrop);
                            maskPredictionImage = addBorder(maskPredictionImage,1,0);
%                             maskPredictionImage(maskPredictionImage>.8) = 1;
%                             maskPredictionImage(maskPredictionImage<.2) = 0;
                            figure(1)
                            clf;
                            vl_tightsubplot(2,2,1); imagesc2(I);title('input')
                            vl_tightsubplot(2,2,2); imagesc2(sc(cat(3,maskPredictionImage,I),'prob_jet'));
                            
                                                                                                                                        
                            title('shape probability');
                            vl_tightsubplot(2,2,3);
%                             maskPredictionImage_1 = superpixelize(I,maskPredictionImage);
                            
                            
                            gc_segResult = getSegments_graphCut_2(I,maskPredictionImage,[],1);
%                             gc_segResult = local_segmentation(I,maskPredictionImage);
%                             pausre;continue;
                           
                            
                            % find the centroid + majority of mass for the
                            % saliency map.
                            [ii,jj,v] = find(maskPredictionImage_orig);
                            ptsCentroid = centroid(jj,ii,v);
%                             figure,,imagesc2(maskPredictionImage_orig);
%                             plotPolygons(ptsCentroid,'g+');
                            [radii,sums,dists] = cummass(maskPredictionImage_orig,ptsCentroid);                            
                            [yy,xx] = find(dists <= radii(find(sums > .9  ,1,'first')));
                            salBox = pts2Box(xx,yy);
                            %I_sal = I_orig;
                            I_sal = cropper(I_orig,salBox);                                                        
                            I_sal = I_orig;
                             opts.show = false;
                            maxImageSize = 200;
                            opts.maxImageSize = maxImageSize;
                            spSize = 90;
                            opts.pixNumInSP = spSize;
                            conf.get_full_image = true;
                            I_sal = imResample(I_sal,size(sal),'bilinear');
                            [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I_sal),opts);
%                           
                            vl_tightsubplot(2,2,4);
                            imagesc2(sc(cat(3,sal,im2double(I_sal)),'prob'));

%                             imagesc2(double(repmat(gc_segResult,[1,1,3])).*I);title('graph-cut result');
                            
%                             saveas(gcf,fullfile('/home/amirro/notes/images/2014_09_18',sprintf('%04.0f.jpg',t)));
                            pause;continue;
                            obj_bbox = pMapToBoxes(objPredictionImage,50,1);                                                                                    
                            %                          clf;imagesc2(I); plotBoxes(obj_bbox);pause;continue;
                            %                         obj_bbox = L.boxes_orig(1,1:4); % translate to coordinate system of roiBox
                            %                         obj_bbox = (obj_bbox-roiBox([1 2 1 2]))*scaleFactor;
                            gt_mask = poly2mask2(box2Pts(obj_bbox),size2(I));
%                                                     clf;imagesc2(I); plotBoxes(obj_bbox);pause;continue;
                            gt_mask = cropper(gt_mask,round(mouth_bbox));
                        end
                        if (nnz(real_gt_mask)==0)
                            fra_db(t).isValid = false;
                            continue
                        end
                        if (squareMask)
                            region_center = boxCenters(region2Box(gt_mask));
                            region_box = [region_center region_center];
                            region_box = inflatebbox(region_box,[70 70],'both',true);
                            gt_mask = poly2mask2(box2Pts(region_box),size2(gt_mask));
                        end
                        fra_db(t).gt_mask = gt_mask;
                        fra_db(t).mouth_image = cropper(I,round(mouth_bbox));
                         figure(2);clf; displayRegions(fra_db(t).mouth_image,fra_db(t).gt_mask); pause
                        %
                    end
                    % % %     save ~/storage/misc/fra_db_with_gt_masks fra_db
                    
                    % % load ~/storage/misc/fra_db_with_gt_masks;
                end
                
                expResults(iExp).params = expParams;
                
                
                %%
                extractFeats = true;
                debug_ = false;
                if (extractFeats)
                    tt = 0;
                    for u = 1:length(fra_db)
                        if (debug_)
                            t = iu_debug(u)
                        else
                            t = u;
                        end
                        %   25
                        %         if (~isempty(fra_db(t).feats)),continue,end
                        if (~fra_db(t).isValid),continue,end
                        if (fra_db(t).classID==5),continue,end
                        a = fra_db(t).mouth_image;
                        a = imResample(double(a),1*size(a),'bilinear');
                        b = fra_db(t).gt_mask;
                        b = bwmorph(b,'clean');
                        if (nnz(b) == 0)
                            continue;
                        end
                        t
                        %      clf;
                        curFeats = {};
                        %
                        
                        b_props = regionprops(b,'Area','PixelIdxList');
                        areas = [b_props.Area];
                        [~,iMax] = max(areas);
                        b = false(size(b));
                        b(b_props(iMax).PixelIdxList) = true;
                        % retain only the largest component of b, to get read of annoyances.
                        M = bwlabel(b);
                        U = unique(M);U = U(U>0);
                        assert(length(U)==1)
                        ff = {};
                        for ii = 1:length(U)
                            ff{ii} = extractFeaturesForSegment(a,M==U(ii),featureExtractor,debug_);
                            % flip
                            %             if (fra_db(t).isTrain)
                            %                 ff{ii+1} = extractFeaturesForSegment(flip_image(a),flip_image(M==U(ii)),featureExtractor,debug_);
                            %             end
                            if (debug_)
                                pause
                            end
                            tt= tt+1;
                            %             saveas(gcf,['/home/amirro/notes/images/2014_08_06/',num2str(tt) '.png']);
                            %             pause;
                        end
                        
                        for uu =1:length(ff)
                            curFeats = ff{uu};
                            
                            if (any(cellfun(@(x) any(isnan(x(:))),curFeats.shapeFeats)))
                                error('nan shape feats!');
                            end
                            if (any(cellfun(@(x) any(isnan(x(:))),curFeats.appearanceFeats)))
                                error('nan appearance feats!');
                            end
                            
                            if (all(cellfun(@(x) all(x(:)==0),curFeats.shapeFeats)))
                                error('all zero shape features - marking as invalid');
                                fra_db(t).isValid = false;
                            end
                            if (all(cellfun(@(x) all(x(:)==0),curFeats.appearanceFeats)))
                                error('all zeros appearance features - marking as invalid');
                                fra_db(t).isValid = false;
                            end
                            
                            %         curFeats{1}.shapeFeats
                        end
                        if (fra_db(t).isValid)
                            fra_db(t).feats = ff;
                        end
                        %load ~/storage/misc/fra_db_with_feats fra_db
                    end
                    %     save ~/storage/misc/fra_db_with_feats fra_db
                end
                
                %% train and classify.
                
                dostuff2 = true;
                if (dostuff2)
                    
                    all_class_ids = [fra_db.classID];
                    train_set = [fra_db.isTrain];
                    res = struct;
                    all_feats1 = {};
                    all_feats2 = {};
                    all_feats3 = {};
                    all_feats6 = {};
                    appearanceFeats = {};
                    all_feats7 = {};
                    touchingMouth = zeros(size(fra_db));
                    touchingMouthCenter = zeros(size(fra_db));
                    for t = 1:length(fra_db)
                        if (isempty(fra_db(t).feats))
                            fra_db(t).isValid = false;
                        end
                    end
                    
                    for t = 1:length(fra_db)
                        if (~fra_db(t).isValid),continue,end
                        
                        curFeats = fra_db(t).feats{1};
                        
                        touchingMouth(t) = sum(curFeats.shapeFeats{1}) > 0;
                        touchingMouthCenter(t) = curFeats.shapeFeats{1}(5);
                        shapeFeats = cellfun2(@col,curFeats.shapeFeats);
                        shapeFeats = cat(1,shapeFeats{:});
                        %appearanceFeats = cellfun2(@(x) x(:)/sum(x(:).^2)^.5,curFeats.appearanceFeats);
                        appearanceFeats = cellfun2(@(x) x(:),curFeats.appearanceFeats);
                        %         if (any(isnan(cat(1,shapeFeats{:}))))
                        %             error('nan shape feats!');
                        %         end
                        %         if (any(isnan(cat(1,appearanceFeats{:}))))
                        %             error('nan shape feats!');
                        %         end
                        
                        %         appearanceFeats{t} =
                        %all_feats{t} = [shapeFeats;appearanceFeats];
                        if rotateObj
                            all_feats1{t} = cat(1,appearanceFeats{[2 3 6]});
                        else
                            all_feats1{t} = cat(1,appearanceFeats{[2 3]});
                        end
                        %                 all_feats1{t} = cat(1,appearanceFeats{[1]});                        
                        %         all_feats1{t} = cat(1,appearanceFeats{[1 2]});                        
                        %         all_feats2{t} = cat(1,appearanceFeats{[2]});
                        %         all_feats3{t} = cat(1,appearanceFeats{[3]});
                        %         all_feats4{t} = cat(1,appearanceFeats{[4]});
                        %         all_feats5{t} = cat(1,appearanceFeats{[5]});
                        %         all_feats6{t} = cat(1,appearanceFeats{[6]});
                        all_feats7{t} = shapeFeats;
                        %          all_feats{t} = appearanceFeats(1:5,:);
                        %          all_feats{t} = single(all_feats{t});
                    end
                    getSizes = @(t) cellfun(@(y) size(y,1),t);
                    goodSizes = @(t) getSizes(t) == mode(getSizes(t));
                    valids = [fra_db.isValid];
                    valids = valids & goodSizes(all_feats1) & goodSizes(all_feats7);% & goodSizes(all_feats3) & goodSizes(all_feats6);
                    % %
                    %     sizes = cellfun(@(x) size(x,1),all_feats);
                    %     valids = valids & sizes==max(sizes);
                    %     sizes = cellfun(@(x) size(x,1),all_feats2);
                    %     valids = valids & sizes==max(sizes);
                    %     X = cat(2,all_feats{valids});
                    %     [d,mu,stds] = normalizeData(X);
                end
                
                
                %%
                n = 0;
                excludeClass = 5;
                aucs = zeros(1,4);
                recs = {};precs = {};
                for iClass =1:4;%length(classes)
                    n = n+1;
                    res(n).className = classNames{iClass};
                    res(n).class_id = iClass;
                    sel_class = all_class_ids==iClass;
                    res2 = classificationHelper(all_feats1,train_set,valids,iClass,all_class_ids,excludeClass,false);
                    res7 = classificationHelper(all_feats7,train_set,valids,iClass,all_class_ids,excludeClass,true);
                    sel_test = ~train_set & valids & (all_class_ids~=excludeClass);
                    %ims = {fra_db(sel_test).imageID};
                    ims = {fra_db(sel_test).mouth_image};
                    test_labels = [fra_db(sel_test).classID];
                    test_scores = res2+0*1*res7;
                    close all;
                    figure('name',classNames{iClass});
                    [u,iu] = sort(test_scores,'descend');
                    ims = paintRule(ims,test_labels==iClass,[0 0 0],[255 0 0],4);
                    nnz(test_labels==iClass)
                    MM = mImage(ims(iu(1:100)));
                    
                    scores_debug = -inf(size(sel_test));
                    scores_debug(sel_test) = test_scores;
                    scores_debug(sel_class) = -inf;
                    [u_debug,iu_debug] = sort(scores_debug,'descend');
                    %                     figure(1)
                    %                     vl_tightsubplot(1,2,1); imagesc2(MM);
                    %                     vl_tightsubplot(1,2,2);
                    %                     vl_pr(2*(test_labels==iClass)-1, test_scores,'IncludeInf',false);
                    [rec,prec,info] = vl_pr(2*(test_labels==iClass)-1, test_scores,'IncludeInf',false);
                    recs{iClass} = rec;
                    precs{iClass} = prec;
                    aucs(iClass) = info.auc;
                    
                    %     figure(2),
                    %     test_scores = test_scores+100*double(col(touchingMouthCenter(sel_test)));
                         vl_pr(2*(test_labels==iClass)-1, test_scores,'IncludeInf',false);
                                    pause
                    %     se
                    t(gcf,'units','normalized','outerposition',[0 0 1 1]);
                    %         saveas(gcf,['/home/amirro/notes/images/2014_08_10/' classNames{iClass} '.png']);
                    %     [u,iu] = sort(test_scores,'descend');
                    %     displayImageSeries(conf,ims(iu));
                    
                end
                expResults(iExp).classifierResults.aucs = aucs;
                expResults(iExp).classifierResults.recs = recs;
                expResults(iExp).classifierResults.precs = precs;
                expResults(iExp).iClass = iClass;
            end
        end
    end
end

% aggregate all the results

iExp = 0;
summary = zeros(4,length(expResults),6);
% n = 0;
for mouthKnown = mouthKnowns
    for objectKnown = objectKnowns
        for squareMask = squareMasks
            for rotateObj = rotateObjs
                iExp = iExp + 1;
                for iClass =1:4
                    %                 n = n+1;
                    summary(iClass,iExp,:) = [iClass mouthKnown objectKnown squareMask rotateObj expResults(iExp).classifierResults.aucs(iClass)];
                end
            end
        end
    end
end

MOUTH = 1;
OBJECT = 2;
SQUARE_MASK = 3;
ROT_OBJ = 4;
%%
for iClass = 1:4
    curClassResults = squeeze(summary(iClass,:,:));
    [u,iu] = sort(curClassResults(:,end),'descend');
    curClassResults = curClassResults(:,2:end);
    %disp(curClassResults(iu,:));
    sel_false= ~curClassResults(:,SQUARE_MASK);
    sel_true = ~sel_false;
    disp(classNames{iClass});
    fprintf('\tmouth\tobject\tsquare_obj\trot_obj\n');
    disp(curClassResults(:,:));
    disp(['off : ' num2str(mean(curClassResults(sel_false,end)))]);
    disp(['on : ' num2str(mean(curClassResults(sel_true,end)))]);    
    pause
end

% save ~/storage/misc/summary_no_rotated_obj.mat summary
