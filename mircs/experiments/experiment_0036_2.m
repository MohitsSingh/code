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
% addpath(genpath('/home/amirro/code/3rdparty/spagglom_01'));
featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];

%%
specific_classes_init;
%%
all_feats = struct('feats',{},'name',{},'isValid',{},'extra',{});

%% extract sub images and corresp . features
nFeat = 0;
%%
feat_data_path = '~/storage/misc/baseLine_feats_12.mat';
% if (exist(feat_data_path,'file'))
%     load(feat_data_path);
% else

%% initialize alternative faces...
% validIndices = validIndices(~isTrain);
% validIndices = validIndices(1:100);
validFaces = true(size(validIndices));
%%
for t = 1:length(validIndices)
    curImageData = newImageData(validIndices(t));
    
    dataInd = findImageIndex(data,curImageData.imageID);
    
    L = load(j2m('~/storage/s40_upper_body_faces/',curImageData.imageID));
    if (~isempty(L.res)) % found a face...
        if (isempty(data(dataInd).upperBodies))
            newImageData(validIndices(t)).faceScore = -1000;
        else
            newImageData(validIndices(t)).alternative_face = L.res(1,1:4)+data(dataInd).upperBodies([1 2 1 2]);
        end
    else
        newImageData(validIndices(t)).faceScore=-1000;
    end
    ub = data(dataInd).upperBodies;
    %          [w h a] = BoxSize(round(ub));
    %         pred_xy = gmfit.mu(1:2).*[w h]+ub(1:2);
    %         curSub = data(ik).subs{1};
    % %     resizeFactor = 128/size(curSub,1);
    % %     curSub = imResample(curSub,resizeFactor);
    %     pred_s = gmfit.mu(3)*(w+h)/2;
    %         break
    %     end
end

%%
face_images = {};
ticId = ticStatus('face images',1,.1);
for k = 1:length(validIndices)
    curImageData = newImageData(validIndices(k));
    [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1,false,use_manual_faces);
    face_images{k} = M;
    tocStatus( ticId, k/length(isTrain));
end
fprintf('\n');
mImage(face_images);
%%
% upper body detections - a proxy for face detection.
%
obj_images = {};
ticId = ticStatus('object images',1,.1);
for k = 1:length(isTrain)
    obj_bbox = newImageData(validIndices(k)).obj_bbox;
    %obj_bbox = round(inflatebbox(obj_bbox,1.1,'both',false));
    [I,I_rect] = getImage(conf, newImageData(validIndices(k)));
    %         clf; imagesc2(I); hold on; plotBoxes(obj_bbox,'g--','LineWidth',2);
    obj_bbox = round(obj_bbox);
    M = cropper(I,obj_bbox);
    if (size(M,1) < 100)
        M = imresize(M,[100 NaN],'bilinear');
    end
    obj_images{k} = M;
    %         [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,newImageData(validIndices(k)),.5,true);
    %         mouth_images{k} = M;
    tocStatus( ticId, k/length(isTrain));
end

resizeX = @(X) cellfun2(@(x) imResample(x,[120 120],'bilinear'),X);
face_images = resizeX(face_images);
%     mouth_images = resizeX(mouth_images);
allFeats_face = extractFeatures_(featureExtractor,face_images);

nFeat = nFeat+1;
all_feats(nFeat).feats = allFeats_face;
all_feats(nFeat).isValid = validFaces & true(1,size(allFeats_face,2));
all_feats(nFeat).name = 'fisher_face';
all_feats(nFeat).shortname = 'Face';

allFeats_obj = extractFeatures_(featureExtractor,obj_images);
nFeat = nFeat+1;
all_feats(nFeat).feats = allFeats_obj;
all_feats(nFeat).isValid = true(1,size(allFeats_face,2));
all_feats(nFeat).name = 'fisher_obj';
all_feats(nFeat).shortname = 'Obj';

% just use the person bounding boxes in all images...
feats_global = {};
feats_saliency = {};
ticId = ticStatus('Global features',1,.5);
for k = 1:length(isTrain)
    tocStatus(ticId,k/length(isTrain));
    curImageData = newImageData(validIndices(k));
    [I,I_rect] = getImage(conf, newImageData(validIndices(k)));
    I = cropper(I,I_rect);
    I = imResample(I,256/size(I,1),'bilinear');
    feats_global{k} = featureExtractor.extractFeatures(I);
end

ticId = ticStatus('Saliency features',1,.5);
for k = 1:length(isTrain)
    tocStatus(ticId,k/length(isTrain));
    curImageData = newImageData(validIndices(k));
    [I,I_rect] = getImage(conf, newImageData(validIndices(k)));
    T = .55;
    sMap = foregroundSaliency(conf,curImageData.imageID);
    saliencyMap = sMap>T;
    while (nnz(saliencyMap)==0)
        T = T-.1;
        saliencyMap = sMap > T;
    end
    %     T
    [yy,xx] = find(saliencyMap);
    obj_bbox = round(pts2Box([xx yy]));
    M = cropper(I,obj_bbox);
    S = cropper(saliencyMap,obj_bbox);
    if (size(M,1) > 100)
        M = imresize(M,[100 NaN],'bilinear');
        S = imresize(S,[100 NaN],'bilinear');
    end
    feats_saliency{k} = featureExtractor.extractFeatures(M,S>0);
end
nFeat = nFeat+1;

all_feats(nFeat).feats = cat(2,feats_saliency{:});
all_feats(nFeat).name = 'feats_saliency';
all_feats(nFeat).isValid = true(1,size(all_feats(nFeat).feats,2));
all_feats(nFeat).shortname = 'Sal';

nFeat = nFeat+1;
all_feats(nFeat).feats = cat(2,feats_global{:});
all_feats(nFeat).name = 'feats_global';
all_feats(nFeat).isValid = true(1,size(all_feats(nFeat).feats,2));
all_feats(nFeat).shortname = 'Global';

% action region prediction
% NOTE: we're using for ground truth regions the same regions as action-obj
ticId = ticStatus('action region prediction',1,.5);
action_region_feats = {};
featureExtractor.bowConf.bowmodel.numSpatialX = [1];
featureExtractor.bowConf.bowmodel.numSpatialY = [1];

for k = 1:length(isTrain)
    
    if (size(action_region_feats{k},1)<16000)
        continue;
    end
        
    
    k
%     if (size(action_region_feats{k},1)<17000),continue;end
    if (isTrain(k))
        action_region_feats{k} = allFeats_obj(1:16384,k);
        continue;
    end
%     break
%     tocStatus(ticId,k/length(isTrain));
    curImageData = newImageData(validIndices(k));
    [I,I_rect] = getImage(conf, newImageData(validIndices(k)));
    load (j2m(conf.action_pred_dir,curImageData.imageID));
    Q = imResample(Q,size2(I));Q = Q-min(Q(:));Q = Q/max(Q(:));    
%     clf; subplot(1,2,1);imagesc2(I);subplot(1,2,2);imagesc2(Q);pause;continue
    action_region_feats{k} = featureExtractor.extractFeatures(I,Q>0.3);
end

% 
% s = cellfun(@(x) size(x,1),action_region_feats)
% 
% for k = 1:length(isTrain)
%     if (isTrain(k))
%         action_region_feats{k} = allFeats_obj(1:16384,k);
%     end
% end

%16384

nFeat = nFeat+1;
all_feats(nFeat).feats = cat(2,action_region_feats{:});
all_feats(nFeat).name = 'feats_action';
all_feats(nFeat).isValid = true(1,size(all_feats(nFeat).feats,2));
all_feats(nFeat).shortname = 'Action_pred';

% all_feats(nFeat).extra = masks_kp;

% % ticId = ticStatus('upper body images',1,.1);
% % upper_body_images = {};
% % feats_upper_body = {};
% % for k = 1:length(isTrain)
% %     curSub = data(k).subs;
% %     if isempty(curSub)
% %         curSub = zeros(128,128,3);
% %     else
% %         curSub = imResample(curSub{1},128/size(curSub{1},1),'bilinear');
% %     end
% %     upper_body_images{k} = curSub;
% %     feats_upper_body{k} = featureExtractor.extractFeatures(curSub);
% %     tocStatus( ticId, k/length(isTrain));
% % end
% % 
% % 
% % nFeat = nFeat+1;
% % all_feats(nFeat).feats = cat(2,feats_upper_body{:});
% % all_feats(nFeat).name = 'feats_upper_body';
% % all_feats(nFeat).isValid = true(1,size(all_feats(nFeat).feats,2));
% % all_feats(nFeat).shortname = 'up.bod';


%%
%%
save(feat_data_path,'all_feats','nFeat','-v7.3');

%%
%     save(feat_data_path,'face_images','mouth_images','allFeats_face','allFeats_mouth','masks','maskedFeats','face_masks');
% end
%% train on different feature types
res = struct('className',{},'class_id',{},'feat_id',{},'recall',{},'precision',{},'info',{},'classifier',{},...
    'feat_name',{});
%%
n = 0;
learnWeights = false;
if (learnWeights)
    sel_factor = 2;
else
    sel_factor = 1;
end

for iFeatType = 1:length(all_feats)
    
    curFeats = all_feats(iFeatType).feats;
    valids = all_feats(iFeatType).isValid;
    for iClass = 1:length(classes)
        n = n+1;
        if (iFeatType~=5),continue,end;
        res(n).className = classNames{iClass};
        res(n).class_id = iClass;
        curLabel = class_labels==iClass;
        poss = find(curLabel == 1 & isTrain & valids);
        negs = find(~curLabel & isTrain & valids);
        features_pos = curFeats(:,poss(1:sel_factor:end));
        features_neg = curFeats(:,negs(1:sel_factor:end));
        
        % leave half for validation.
        
        classifier = train_classifier_pegasos(features_pos,features_neg,-1);
        res(n).classifier.w =           classifier.w;
        res(n).classifier.optAvgPrec =  classifier.optAvgPrec;
        res(n).classifier.optLambda =   classifier.optLambda;
        %
        test_feats = curFeats(:,~isTrain);
        [yhat,h_] = classifier.test(double(test_feats));
        [res(n).recall, res(n).precision, res(n).info] = vl_pr(2*(class_labels(~isTrain)==iClass)-1, h_);
        res(n).feat_name = all_feats(iFeatType).name;
        res(n).feat_id = iFeatType;
    end
end

% now for each class find the best combination of scores.

if (learnWeights)
    all_weights = zeros(length(classes),length(all_feats));
    
    for iClass = 1:length(classes)
        ress = res([res.class_id]==iClass);
        curLabel = class_labels==iClass;
        h_ = {};
        poss = find(curLabel == 1 & isTrain);
        negs = find(~curLabel & isTrain);
        poss = poss(2:2:end);negs = negs(2:2:end);
        nPos = length(poss);
        for iFeatType = 1:length(all_feats)
            curFeats = all_feats(iFeatType).feats;
            features_pos = curFeats(:,poss);
            features_neg = curFeats(:,negs);
            h_{iFeatType} = ress(iFeatType).classifier.w(1:end-1)'*(double([features_pos,features_neg]));
        end
        h_ = cat(1,h_{:});
        classifier = train_classifier_pegasos(h_(:,1:nPos),h_(:,nPos+1:end),-1);
        all_weights(iClass,:) = classifier.w(1:end-1);
    end
end
% use the second half to find the best combination of scores...
% % % % save('~/storage/misc/baseLine_classifiers_12.mat','res');

%%
%% clc;
% A more concise summary
writeOutput = true;
writeImg = false;
tostd = 1;
ensuredir('~/notes/images/2014_06_08');
if writeOutput
    if tostd , fid = 1; else
        fid = fopen('~/notes/images/2014_06_08/res.txt','w');
    end
end
colors = hsv(5);
lineStyles = {'-.','--','-',':'};
markerStyles = {'s','o','d','*'};
%feat_abbr = {'Face','Obj','Global','Face_kp','Mouth'};
feat_abbr = {all_feats.shortname};
methods = {all_feats.name};
scores_all_methods = {};
for iClass = 1:length(classNames)
    iClass
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
%     all_hs(5,:) = 0;
    subsets = allSubsets(length(ress));
    
%     subsets(:,5) = 0;
    %     subsets = [1 1 1 1];
        subsets = subsets(sum(subsets,2)<=2,:);
    weights = ones(length(all_feats),1);
    weights(2) = 5;
    %     weights = all_weights(iClass,:)';
    %     subsets = ones(size(weights'));
    method_s = {};
    ap_s = {};
    mStrings = {};
    s = ceil(size(subsets,1).^(1/3));
    [nColor,nStyle,nMarker] = ind2sub([s s s],1:size(subsets,1));
    for iChoice = 1:size(subsets,1)
        choice = subsets(iChoice,:)';
        curWeights = weights.*choice;
        scores_total = curWeights'*all_hs;
        scores_all_methods{end+1} = scores_total;
        tChoice = nColor(iChoice); tStyle = nStyle(iChoice);tMarker = nMarker(iChoice);
        sel_ = ~isTrain & class_labels ~=0;
        test_labels = class_labels(sel_);
        test_scores = scores_total(sel_);
        test_scores = test_scores + rand(size(test_scores))*.01;
        [recall, precision, info] = vl_pr(2*(test_labels==iClass)-1, test_scores);
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
        %ir  = ir(1:min(3,length(ir)));
%         ir  = ir([1:3 end-2:end]);
    pad_size = 10;
    padel_string = sprintf('%%-%ds',pad_size);
    padel_f = sprintf('%%-%d.3f',pad_size);
    if (writeOutput)
        fprintf(fid,'\n%s:\n',curClassName);
        for t = 1:nFeat
            fprintf(fid,padel_string,feat_abbr{t});
        end;
        fprintf(fid,'\n');
        fprintf(fid,'%s\n',repmat('-',1,pad_size*(nFeat+1)));
        %         ir = 1:length(aps)        
        v = '-+';        
        for im = 1:length(ir)
            m = ir(im);
            for t=1:nFeat
                fprintf(fid,'%-10s',v(subsets(m,t)+1));
            end
            fprintf(fid,[padel_f '\n'],aps(m));
        end
        %         set(gcf,'units','normalized','outerposition',[0 0 .7 .7]);
        if (writeImg)
            export_fig(sprintf('/home/amirro/notes/images/2014_06_08/%s.pdf',curClassName));
        end
    end
    if (tostd)
        pause;
    end
end
fclose all;
