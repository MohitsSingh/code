%%%%%%% Experiment 0036_3 %%%%%%%%%%%
%%%%%%% May 28, 2014 %%%%%%%%%%%%%%

%% learn, by iterating, a feature representation which is a combination of features
% from different facial regions.

%% initialization
default_init;
addpath(genpath('/home/amirro/code/3rdparty/spagglom_01'));
trainingData = getTrainingPatches(conf,imageData,newImageData,true); % train_ims, rects, face_rects
extractors = initializeFeatureExtractors(conf);
% featureExtractor = BOWFeatureExtractor(conf,conf.featConf(1)); % use all feature types.
featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
featureExtractor.bowConf.bowmodel.numSpatialX = [1];
featureExtractor.bowConf.bowmodel.numSpatialY = [1];
% %%
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
isValid = [newImageData.faceScore] > -.6;
isTrain = [newImageData.isTrain] & isValid;

% for non-valid faces, check if there is a ground truth annotation
% and add it. 
%img_sel = false(size(newImageData)); img_sel(class_labels>0) = true;
img_sel = isValid;
faceActionImageNames = imageNames(img_sel);
save faceActionImageNames faceActionImageNames isValid;
isTrain_ = isTrain; isValid_ = isValid;
isTrain = isTrain(isValid);
class_labels = class_labels(isValid);
validIndices = find(isValid);
all_feats = struct('feats',{},'name',{},'isValid',{},'extra',{});

%% extract sub images and corresp . features
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
allFeats_face = extractFeatures_(featureExtractor,face_images);
allFeats_mouth = extractFeatures_(featureExtractor,mouth_images);
% now extract features by masking all regions except the ones covered
% by the piotr keypoints.
% assume rcp1 has been applied to these faces

all_feats(1).feats = allFeats_face;
all_feats(1).isValid = true(1,size(allFeats_face,2));
all_feats(1).name = 'fisher_face';
all_feats(2).feats = allFeats_mouth;
all_feats(2).isValid = true(1,size(allFeats_mouth,2));
all_feats(2).name = 'fisher_mouth';
L_xy = load('~/storage/misc/face_images_1_xy.mat');
%%
debug_ = false;
ticId = ticStatus('kp masks',1,.5);
% group regions by location, so you won't get a monstrous feature
% vector...
sel_groups = {conf.piotr_coords.eye_left,...
    conf.piotr_coords.eye_right,...
    conf.piotr_coords.nose,...
    conf.piotr_coords.mouth,...
    conf.piotr_coords.chin};
nGroups = length(sel_groups);
feats_masked_kp = cell(length(isTrain),nGroups);
masks_kp = cell(length(isTrain),nGroups);

rad = 25;
% extract a different feature vector for each of the keypoints...
ticId = ticStatus('keypoints features',1,.1);
for k = 1:length(isTrain)
    %         k
    tocStatus(ticId,k/length(isTrain));
    curImageData = newImageData(validIndices(k));
    [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1.5,false);
    curPoly = L_xy.xys{k};
    resizeRatio = 120/size(M,1);
    M = imResample(M,resizeRatio,'bilinear');
    curPoly = round(curPoly*resizeRatio);
    rois = {};
    bad_pts = ~inBox(size2(M),curPoly);
    for iRegion = 1:length(sel_groups)
        point_sel = false(size(curPoly,1),1);
        point_sel(sel_groups{iRegion}) = true;
        point_sel(bad_pts) = false;
        Z = false(size2(M));%                 
%         bb = region2Box(Z);
        bc = mean(curPoly(point_sel,:),1);bc = round(bc);
        Z(sub2ind2(size2(Z),fliplr(bc)))=1;        
        %bc = boxCenters(bb);bc = [bc bc];
%         bc = inflatebbox(bc,size(M,1)/4,'both',true);
        kp_mask = bwdist(Z)<=rad;
%         clf; displayRegions(M,kp_mask); pause;
        masks_kp{k,iRegion} = kp_mask;
        feats_masked_kp{k,iRegion} = col(featureExtractor.extractFeatures(M,kp_mask));                    
    end
end
feats_masked_kp = feats_masked_kp';
% save ~/storage/misc/kp_feats.mat masks_kp feats_masked_kp

%% train on different feature types
res = struct('className',{},'feat_id',{},'recall',{},'precision',{},'info',{},'classifier',{},...
    'feat_name',{});

dim_f = size(feats_masked_kp{1,1},1);
nSamples = length(isTrain);
all_feats = zeros(dim_f*nGroups,nSamples);
for t = 1:nSamples
    all_feats(:,t) = cat(1,feats_masked_kp{:,t});
end

% initialize S = 1
S = ones(nGroups,1);
% S = S/length(S);
nIterations = 5;
valids = true(1,nSamples);

%for iClass = 1:length(classes)
T = 5; % no. iterations.
n = 0;
for iClass = 1
    n = n+1;
    curLabel = class_labels==iClass;
    poss = curLabel == 1 & isTrain & valids;
    negs = ~curLabel & isTrain & valids;
    Ws = {};
    Ss = {}; 
    for t = 1:T
        % stage 1: create weighted feature vector and learn w
        groupWeights = repmat(S',dim_f,1);
        groupWeights = groupWeights(:);
        curFeats = bsxfun(@times,all_feats,groupWeights);
        features_pos = curFeats(:,poss);
        features_neg = curFeats(:,negs);
        classifier = train_classifier_pegasos(features_pos,features_neg,-1);       
        w = classifier.w(1:end-1);
        % stage 2: use <w,x> as new features and learn weighting s
        features_pos_w = zeros(nGroups,size(features_pos,2));
        features_neg_w = zeros(nGroups,size(features_neg,2));        
        for u = 1:nGroups
            block = (u-1)*dim_f+1:u*dim_f;
            features_pos_w(u,:) = w(block)'*features_pos(block,:);
            features_neg_w(u,:) = w(block)'*features_neg(block,:);
        end
        classifier_s = train_classifier_pegasos(features_pos_w,features_neg_w,-1);
        S = classifier_s.w(1:end-1);%S = S/sum(S);
        Ws{t} = w;
        Ss{t} = S;
    end    
end

for u = 1:length(Ws)
    u
    scores_total = Ws{u}'*curFeats;
    [recall, precision, info] = vl_pr(2*(class_labels(~isTrain)==iClass)-1, scores_total(~isTrain));
    clf;plot(recall,precision);pause
    info.ap
end
%         plot(recall,precision);
%%
save ~/mircs/experiments/classifiers res
%%
clc;
% Summarize results, create tables for comparison, do some visualization
% L = load('~/storage/misc/occ_and_more.mat');
writeOutput = true;
writeImg = false;
tostd = false;
if writeOutput
    if tostd , fid = 1; else
        fid = fopen('~/notes/images/2014_05_26/res.txt','w');
    end
end
%colors = {'r','b','g','m','b','y','k'};
colors = hsv(5);
lineStyles = {'-.','--','-',':'};
markerStyles = {'s','o','d','*'};
methods = {all_feats.name};
scores_all_methods = {};

for iClass = 1:length(classNames)
    clf;hold on;
    set(gcf,'DefaultAxesLineStyleOrder',lineStyles)
    curClassName = classNames{iClass};
    ress = res(iClass:length(classNames):end);
    feat_ids = [ress.feat_id];
    all_hs = zeros(length(feat_ids),length(isTrain));
    classifiers = [ress.classifier];
    valids = cat(1,all_feats(feat_ids).isValid);
    for t = 1:length(classifiers)
        w = classifiers(t).w;
        all_hs(t,:) = w(1:end-1)'*all_feats(feat_ids(t)).feats;
    end
    choices = 1:(2^length(ress)-1); % create all combinations of different features types
    B = dec2bin(choices);
    BB = repmat(' ',size(B,1),2*size(B,2));
    BB(:,1:2:end) = B;
    B = str2num(BB);
    weights = ones(length(all_feats),1);
    method_s = {};
    ap_s = {};
    mStrings = {};
    s = ceil(length(choices)^(1/3));
    [nColor,nStyle,nMarker] = ind2sub([s s s],choices);
    for iChoice = 1:length(choices)
        choice = B(iChoice,:)';
        curWeights = weights.*choice;
        scores_total = curWeights'*all_hs;
        scores_all_methods{end+1} = scores_total;
        tChoice = nColor(iChoice); tStyle = nStyle(iChoice);tMarker = nMarker(iChoice);
        [recall, precision, info] = vl_pr(2*(class_labels(~isTrain)==iClass)-1, scores_total(~isTrain));
        plot(recall,precision,'color',colors(tChoice,:),'LineStyle',...
            lineStyles{tStyle},'Marker',markerStyles{tMarker}, 'LineWidth',2);
        curMethods = methods(choice>0);
        mString =[];
        for k = 1:length(curMethods)
            mString = [mString curMethods{k}];
            if (k < length(curMethods))
                mString = [mString ' + '];
            end
        end
        method_s{end+1} = mString;
        ap_s{end+1} = info.ap;
        mString = [mString '(' sprintf('%0.3f',info.ap) ')'];
        mStrings{end+1} = mString;
    end
    xlabel('precision'); ylabel('recall');
    h = get(gca,'Title');
    curTitle=get(h,'String');
    set(h,'interpreter','none');
    %     h = get(gcf,'Legend');
    %     set(gca,);
    legend(mStrings,'interpreter','none');
    title(curClassName);
    aps = cat(1,ap_s{:}); %[r,ir] = sort(aps,'descend');
    %     maximizeFigure;
    [r,ir] = sort(aps,'descend');
    if (writeOutput)
        fprintf(fid,'%s\n',curClassName);
        fprintf(fid,'%55s%55s\n','Method','AP');
        fprintf(fid,'%s\n',repmat('-',1,55*2));
        for im = 1:3%length(aps)
            m = ir(im);
            addString = '';
            tf = m==ir(1:3);
            if (any(tf))
                tf = find(tf);
                addString = repmat('*',[1 tf]);
            end
            fprintf(fid,'%55s%55.3f\n',[method_s{m} addString],aps(m));
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
    end
    pause;
end
fclose all;