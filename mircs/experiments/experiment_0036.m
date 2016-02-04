%%%%%%% Experiment 0036 %%%%%%%%%%%
%%%%%%% May 12, 2014 %%%%%%%%%%%%%%

%% The purpose of this experiment is to construct a baseline for differentiating
% between similar action classes.
% The action classes are drinking, smoking, blowing bubbles, brushing
% teeth.
% The differentiation will be using the head region alone.
% This is to serve as a baseline to show further , more fine grained
% processing is necessary.

%% initialization
default_init;
addpath(genpath('/home/amirro/code/3rdparty/spagglom_01'));
fisherFeatureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
fisherFeatureExtractor.bowConf.bowmodel.numSpatialX = [1];
fisherFeatureExtractor.bowConf.bowmodel.numSpatialY = [1];
%%

% find class labels in dataset
classes = [conf.class_enum.DRINKING;...
    conf.class_enum.SMOKING;...
    conf.class_enum.BLOWING_BUBBLES;...
    conf.class_enum.BRUSHING_TEETH];
classNames = conf.classes(classes);

imageNames={newImageData.imageID};
class_labels = zeros(1,length(newImageData));
for iClass = 1:length(classes)
    isClass = strncmp(classNames{iClass},imageNames,length(classNames{iClass}));
    class_labels(isClass) = iClass;
end
% extract features from valid images
isTrain = [newImageData.isTrain];
isValid = [newImageData.faceScore] >-.6 & class_labels > 0;
img_sel = false(size(newImageData)); img_sel(class_labels>0) = true;
faceActionImageNames = imageNames(img_sel);
save faceActionImageNames faceActionImageNames img_sel isValid;
isTrain_ = isTrain; isValid_ = isValid;
isTrain = isTrain(isValid);
class_labels = class_labels(isValid);
validIndices = find(isValid);



%% extract sub images and corresp . features

feat_data_path = '~/storage/misc/baseLine_feats.mat';

if (exist(feat_data_path,'file'))
    load(feat_data_path);
else
    face_images = {};
    ticId = ticStatus([],1,.1);
    for k = 1:length(isTrain)
        [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,newImageData(validIndices(k)),1,false);
        face_images{k} = M;
        [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,newImageData(validIndices(k)),.5,true);
        mouth_images{k} = M;
        tocStatus( ticId, k/length(isTrain));
    end
    resizeX = @(X) cellfun2(@(x) imResample(x,[120 120],'bilinear'),X);
    
    face_images = resizeX(face_images);% cellfun2(@(x) imResample(x,[120 120],'bilinear'),face_images);
    mouth_images = resizeX(mouth_images);%cellfun2(@(x) imResample(x,[120 120],'bilinear'),mouth_images);
    allFeats_face = extractFeatures_(fisherFeatureExtractor,face_images);
    allFeats_mouth = extractFeatures_(fisherFeatureExtractor,mouth_images);
    
    
    % now extract features by masking all regions except the ones covered
    % by the piotr keypoints.
    % assume rcp1 has been applied to these faces
    L_xy=load('~/storage/misc/face_images_1_xy.mat');
    %%
    debug_ = false;
    feats_masked_kp = cell(size(isTrain));
    masks_kp = cell(size(isTrain));
    ticId = ticStatus('kp masks',1,.5)
    for k = 1:length(isTrain)
        tocStatus(ticId,k/length(isTrain));
        curImageData = newImageData(validIndices(k));
        [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1.5,false);
        curPoly = L_xy.xys{k};
        %Z = zeros(size2(M));
        resizeRatio = 120/size(M,1);
        M = imResample(M,resizeRatio,'bilinear');
        curPoly = round(curPoly*resizeRatio);
        clf; imagesc2(M); hold on;
        point_sel = 1:size(curPoly,1);
        rad = 10;
        Z = false(size2(M));
        Z(sub2ind2(size2(Z),fliplr(curPoly(point_sel,:)))) = 1;
        kp_mask = bwdist(Z)<=rad;
        feats_masked_kp{k} = col(fisherFeatureExtractor.extractFeatures(M,kp_mask));
        if (debug_)
            clf;displayRegions(M,kp_mask);
            %         face_poly = bsxfun(@minus,face_poly,face_box([1 2]));
            %         plotPolygons(curPoly,'g+');
            %         plotPolygons(face_poly,'rd');
            %showCoords(curPoly);
            pause
        end
    end
    %%
    %     face_images_1 = {};
    %     for k = 1:length(isTrain)
    %         k
    %         face_images_1{k} = getSubImage(conf,newImageData(validIndices(k)),1.5,false);
    %     end
    %     save ~/storage/misc/face_images_1 face_images_1
    %
    %% extract features from putative occluded regions....
    %     masks = cell(nnz(isValid),1);
    masks = cell(1,length(validIndices));
    %%
    
    t = 0;
    conf.get_full_image = true;
    occPath = conf.occludersDir;
    maskedFeats = cell(1,length(validIndices));
    valid_masked = false(size(isTrain));
    %     for k = 1:length(isTrain)
    %         k
    %         curMask = masks{k};
    %
    %     end
    
    masks = cell(1,length(validIndices));
    masksPath = '~/storage/misc/my_masks.mat';
    occPath = '~/storage/occluders_s40_new_6';
    if (~exist(masksPath,'file'))
        for k = 1:length(validIndices)
            %     k
            validIndices(k)
            curImageData = newImageData(validIndices(k));
            [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1,true);
            ff = j2m(occPath, curImageData);
            I = getImage(conf,curImageData);
            moreFun = @() plotPolygons(face_poly,'g--');
            load(ff);
            
            %             curImageData.occlusionPattern = L.occlusionPattern;
            %             curImageData.mouth_poly = mouth_poly;
            %             curImageData.face_poly = face_poly;
            %             [occludingRegions,occlusionPatterns,region_scores] = getOccludingCandidates_3(conf,I,curImageData);
            if (isempty(occludingRegions))
                valid_masked(k) =false;
                %                 clf;imagesc2(I);
                %                 disp('no valid masks found');pause;
                %                 clc
                continue;
            end
            [r,ir] = sort(region_scores,'descend');
            %ir = ir(1:min(3,length(ir)));
            ir = ir(1:min(1,length(ir)));
            RR = max(cat(3,occludingRegions{ir}),[],3);
            valid_masked(k) = nnz(RR)>25;
            %             t = t+1
            masks{k} = RR;
            %             clf; displayRegions(I,RR);pause
        end
        save(masksPath,'masks');
    else
        load(masksPath);
    end
    
    for k = 1:length(validIndices)
        k
        if (~valid_masked(k)),continue,end;
        [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,newImageData(validIndices(k)),1.5,false);
        curMask = cropper(masks{k},round(face_box));
        curFeats = col(fisherFeatureExtractor.extractFeatures(M,curMask));
        maskedFeats{k} = col(fisherFeatureExtractor.extractFeatures(M,curMask));
    end
    face_masks = cell(1,length(validIndices));
    for k = 1:length(validIndices)
        k
        [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,newImageData(validIndices(k)),1.5,false);
        face_masks{k} = cropper(masks{k},round(face_box));
    end
    %%
    save(feat_data_path,'face_images','mouth_images','allFeats_face','allFeats_mouth','masks','maskedFeats','face_masks');
end

%%
res = struct('className',{},'feature_weights',{},'recall',{},'precision',{},'info',{},'classifier',{});
useFeat = [0 1];
n = 0;
toDisplay = false;
for iUseFace = 1:length(useFeat)
    for iUseMouth = 1:length(useFeat)
        weights = [useFeat(iUseFace) useFeat(iUseMouth)];
        if (all(weights==00))
            continue;
        end
        allFeats = [allFeats_face*useFeat(iUseFace);...
            allFeats_mouth*useFeat(iUseMouth)];
        for iClass = 1:length(classes)
            n = n+1;
            res(n).className = classNames{iClass};
            res(n).feature_weights = weights;
            curLabel = class_labels==iClass;
            poss = curLabel == 1 & isTrain;
            negs = ~curLabel & isTrain;
            features_pos = allFeats(:,poss);
            features_neg = allFeats(:,negs);
            classifier = train_classifier_pegasos(features_pos,features_neg,-1);
            res(n).classifier.w =           classifier.w;
            res(n).classifier.optAvgPrec =  classifier.optAvgPrec;
            res(n).classifier.optLambda =   classifier.optLambda;
            %
            test_feats = allFeats(:,~isTrain);
            [yhat,h_] = classifier.test(double(test_feats));
            [res(n).recall, res(n).precision, res(n).info] = vl_pr(2*(class_labels(~isTrain)==iClass)-1, h_);
            if (toDisplay)
                %                 test_images = face_images(~isTrain);
                clf;vl_pr(2*(class_labels(~isTrain)==iClass)-1, h_);
                h = get(gca,'Title');
                curTitle=get(h,'String');
                set(h,'interpreter','none');
                title([classNames{iClass} ', ' curTitle]);
                pause;
            end
        end
    end
end

%% masked features classification
res_masked = struct('className',{},'feature_weights',{},'recall',{},'precision',{},'info',{},'classifier',{});
n=0;
weights = [0 0];
toDisplay = false;
f_valid = find(valid_masked,1,'first');
for u = 1:length(maskedFeats) % fill in invalid features with 0 for convenience
    if (~valid_masked(u))
        maskedFeats{u} = zeros(size(maskedFeats{f_valid}));
    end
end
allFeats_masked = cat(2,maskedFeats{:});
for iClass = 1:length(classes)
    n = n+1;
    res_masked(n).className = classNames{iClass};
    res_masked(n).feature_weights = weights;
    curLabel = class_labels==iClass;
    poss = curLabel == 1 & isTrain & valid_masked;
    negs = ~curLabel & isTrain & valid_masked;
    features_pos = allFeats_masked(:,poss);
    features_neg = allFeats_masked(:,negs);
    classifier = train_classifier_pegasos(features_pos,features_neg,1);
    res_masked(n).classifier.w =           classifier.w;
    res_masked(n).classifier.optAvgPrec =  classifier.optAvgPrec;
    res_masked(n).classifier.optLambda =   classifier.optLambda;
    %
    test_feats = allFeats_masked(:,~isTrain);
    [yhat,h_mask] = classifier.test(allFeats_masked(:,~isTrain));
end
%%
save ~/mircs/experiments/classifiers_masked res_masked

%%
save ~/mircs/experiments/classifiers res

%%
infos = [res.info];
weights = cat(1,res.feature_weights);
[weights cat(1,infos.ap)]


%%

seg_scores = get_putative_occluders(conf,curImageData,cur_occ,I);
displayRegions(I,cur_occ.occludingRegions,seg_scores,0,5,moreFun);
%     pause

%%

%%
% Summarize results, create tables for comparison, do some visualization
L = load('~/storage/misc/occ_and_more.mat');
regions = {'face','mouth','kp'};
writeOutput = true;
writeImg = false;
tostd = true;
if writeOutput
    if tostd , fid = 1, else
        fid = fopen('~/notes/images/2014_05_18/res.txt','w');
    end
end
colors = {'r','b','g','m','b','y','k'};
region_choice = {'-.','--','-'};
methods = {'fisher','occlusion','dpm','mask'};
choices = [1 0 0;1 0 1;1 1 1]>0; % different combinations of region choices...
choices = [[choices,zeros(3,1)];[choices,ones(3,1)]]>0;
choices = [choices;true true false true];

for iClass = 1:length(classNames)
    clf;
    curClassName = classNames{iClass};
    ress = res(iClass:4:end);
    h_mask =res_masked(iClass).classifier.w(1:end-1)'*allFeats_masked(:,~isTrain)+res_masked(iClass).classifier.w(end);
    h_mask(~valid_masked(~isTrain)) = 0;
    region_s = {};
    method_s = {};
    ap_s = {};
    scores_all_methods = {};
    mStrings = {};
    for iRes = 1:3
        res_ = ress(iRes);
        tString =[];
        curRegions = regions(res_.feature_weights>0);
        for k = 1:length(curRegions)
            tString = [tString curRegions{k}];
            if (k < length(curRegions))
                tString = [tString ' + '];
            end
        end
        curLabel = class_labels==iClass;
        poss = curLabel == 1 & isTrain;
        negs = ~curLabel & isTrain;
        test_feats = [allFeats_face;allFeats_mouth];
        test_feats = test_feats(:,~isTrain);
        h_fisher =res_.classifier.w(1:end-1)'*double(test_feats)+res_.classifier.w(end);
        sel_ = ~isTrain_ & isValid;
        scores_phrase = L.phraseScores(sel_);
        scores_occ = L.occlusionScores(sel_);
        scores_occ(isinf(scores_occ)) = min(scores_occ(~isinf(scores_occ)));
        scores_dpm = L.s_dpm(sel_)';
        weights = [1 .1 1 .5]';
        hold on;
        
        for iChoice = 1:size(choices,1)
            %choice = [1 1 1]';
            choice = choices(iChoice,:)';
            curWeights = weights.*choice;
            allScores = [h_fisher;scores_occ;scores_dpm;h_mask];
            scores_total = curWeights'*allScores;%+.005*straw_scores(isValid & ~isTrain_);
            scores_all_methods{end+1} = scores_total;
            %     vl_pr(2*(class_labels(~isTrain)==iClass)-1, scores_total);
            [recall, precision, info] = vl_pr(2*(class_labels(~isTrain)==iClass)-1, scores_total);
            plot(recall,precision,[colors{iChoice} region_choice{iRes}],'LineWidth',2);
            curMethods = methods(choice);
            mString =[];
            for k = 1:length(curMethods)
                mString = [mString curMethods{k}];
                if (k < length(curMethods))
                    mString = [mString ' + '];
                end
            end
            region_s{end+1} = tString;
            method_s{end+1} = mString;
            ap_s{end+1} = info.ap;
            mString = [tString ';' mString '(' sprintf('%0.3f',info.ap) ')'];
            mStrings{end+1} = mString;
        end
    end
    aps = cat(1,ap_s{:}); %[r,ir] = sort(aps,'descend');
    xlabel('precision'); ylabel('recall');
    legend(mStrings);title(curClassName);
    h = get(gca,'Title');
    curTitle=get(h,'String');
    set(h,'interpreter','none');
    %     maximizeFigure;
    [r,ir] = sort(aps,'descend');
    if (writeOutput)
        fprintf(fid,'%s\n',curClassName);
        fprintf(fid,'%35s%35s%35s\n','Method','Regions','AP');
        fprintf(fid,'%s\n',repmat('-',1,35*3));
        for im = 1:3%length(aps)
            m = ir(im);
            addString = '';
            tf = m==ir(1:3);
            if (any(tf))
                tf = find(tf);
                addString = repmat('*',[1 tf]);
            end
            %fprintf('%s%s%s%s%0.3f\n',method_s1{m},T,region_s1{m},T,r(m));
            fprintf(fid,'%35s%35s%35.3f\n',[method_s{m} addString],region_s{m},aps(m));
        end
        if (writeImg)
            export_fig(sprintf('/home/amirro/notes/images/2014_05_18/%s.pdf',curClassName));
        end
    else
        % show the results for the best classifiers...
        if (0)
            test_images = face_images(~isTrain);
            curScores = scores_all_methods{ir(1)};
            [v,iv] = sort(curScores,'descend');
            m = paintRule(test_images,class_labels(~isTrain)==iClass);
            % create some visualizations : show the mask for each image.
            test_masks = face_masks(~isTrain);
            testIndices = validIndices(~isTrain);
            valid_masked_test = valid_masked(~isTrain);
            for ik = 1:length(testIndices)
                curInd = iv(ik);
                if (~valid_masked_test(iv(ik))),continue;end
                [curFaceImg,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,newImageData(testIndices(curInd)),1.5,false);
                curFaceMask = test_masks{curInd};
                curFaceMask = imResample(curFaceMask,size2(curFaceImg),'nearest');
                faceImgMasked = blendRegion(curFaceImg,curFaceMask);
                face_poly = bsxfun(@minus,face_poly,face_box(1:2));
                clf; imagesc2(faceImgMasked); hold on; plotPolygons(face_poly,'g--');
                pause;continue;
            end
        end
        %         showSorted(m,curScores,50);
    end
    pause
end

fclose all;
