%%%%%%% Experiment 0044 %%%%%%%%%%%
%%%%%%% June 23, 2014 %%%%%%%%%%%%%%

%% The purpose of this experiment is to construct a baseline for differentiating
% between similar action classes, on the fra_db

%% initialization
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
end

featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];

%%
all_feats = struct('feats',{},'name',{},'isValid',{},'extra',{});

%% extract sub images and corresp . features
nFeat = 0;
%%
feat_data_path = '~/storage/misc/fra_db_feats_new.mat';
if (exist(feat_data_path,'file'))
    load(feat_data_path);
else
    %%
    feats_face = {};
    feats_hand = {};
    feats_obj = {};
    ticId = ticStatus('feature extraction',1,.1);
    
    all_feats_2 = struct('feat',{},'type',{},'name','isTrain',{});
    iFeat = 0;
    %TODO: note, for each image, if multiple hand/objects exist, only the
    %last one is used!!!!
    %TODO! add flipped images too later
    for flip = 1
    for k = 1:length(fra_db)
        curImageData = fra_db(k);
        %         I = getImage(conf,curImageData);
        infScale = 2.5;
        [rois,subRect,I] = get_rois_fra(conf,curImageData,infScale);
        for iRoi = 1:length(rois)
            iFeat = iFeat+1;
            curRoi = rois(iRoi);
            curMask = poly2mask2(round(box2Pts(curRoi.bbox)),size2(I));
            all_feats_2(iFeat).isTrain = curImageData.isTrain;
            all_feats_2(iFeat).type = curRoi.id;
            all_feats_2(iFeat).name = curRoi.name;
            all_feats_2(iFeat).feat = featureExtractor.extractFeatures(I,faceBoxRoi);
        end
        
        %         clf; imagesc2(I);
%         plotBoxes(cat(1,rois.bbox));
        faceBox = curImageData.faceBox;
        faceBoxRoi = poly2mask2(box2Pts(round(faceBox)),size2(I));
        feats_face{k} = featureExtractor.extractFeatures(curImageData.imageID,faceBoxRoi);
        handBox = curImageData.hands;
        if (~isempty(handBox)) % maybe no hands participate in action
            handBox = [min(handBox(:,1:2),[],1) max(handBox(:,3:4),[],1)]; % "union" of all boxes
            handBoxRoi = poly2mask2(box2Pts(round(handBox)),size2(I));
            feats_hand{k} = featureExtractor.extractFeatures(curImageData.imageID,handBoxRoi);
        end
        objRoi = false(size2(I));
        if (~isempty(curImageData.objects))
            for ii = 1:length(curImageData.objects) %this is a bug, should use only one
                objRoi = objRoi | poly2mask2(round(curImageData.objects(ii).poly),size2(I));
            end
            feats_obj{k} = featureExtractor.extractFeatures(curImageData.imageID,objRoi);
        end
        tocStatus( ticId, k/length(fra_db));
    end
    end
    fprintf('\n');
       
    valids_face = false(size(fra_db));
    valids_hand = false(size(fra_db));
    valids_obj = false(size(fra_db));
    for t = 1:length(fra_db)
        valids_face(t) = ~isempty(feats_face{t});
        valids_hand(t) =  ~isempty(feats_hand{t});
        valids_obj(t) =  ~isempty(feats_obj{t});
    end    
    valids = valids_face & valids_hand & valids_obj;    
    f_valid = find(valids,1,'first');
    for k = 1:length(fra_db)
        if (valids(k)), continue; end
        feats_face{k} = zeros(size(feats_face{f_valid}));
        feats_hand{k} = zeros(size(feats_hand{f_valid}));
        feats_obj{k} = zeros(size(feats_obj{f_valid}));
    end
        
    nFeat = nFeat+1;
    all_feats(nFeat).feats = cat(2,feats_face{:});
    all_feats(nFeat).name = 'fisher_face';
    all_feats(nFeat).isValid = valids;
    all_feats(nFeat).shortname = 'Face';
        
    nFeat = nFeat+1;
    all_feats(nFeat).feats = cat(2,feats_hand{:});
    all_feats(nFeat).name = 'fisher_hand';
    all_feats(nFeat).isValid = valids;
    all_feats(nFeat).shortname = 'hand';
        
    nFeat = nFeat+1;
    all_feats(nFeat).feats = cat(2,feats_obj{:});
    all_feats(nFeat).name = 'feats_obj';
    all_feats(nFeat).isValid = valids;
    all_feats(nFeat).shortname = 'obj';
    save(feat_data_path,'all_feats','valids','nFeat','-v7.3');
end



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
    curValids = all_feats(iFeatType).isValid & valids;
    for iClass = 1:length(classes)
        n = n+1;
        res(n).className = classNames{iClass};
        res(n).class_id = iClass;
        curLabel = class_labels==iClass;
        poss = find(curLabel == 1 & isTrain & curValids);
        negs = find(~curLabel & isTrain & curValids);
        
        if (any(isnan(curFeats(:))))
            break;
        end
        
        features_pos = curFeats(:,poss(1:sel_factor:end));
        features_neg = curFeats(:,negs(1:sel_factor:end));
        
        % leave half for validation?
        classifier = train_classifier_pegasos(double(features_pos),double(features_neg),-1);
        res(n).classifier.w =           classifier.w;
        res(n).classifier.optAvgPrec =  classifier.optAvgPrec;
        res(n).classifier.optLambda =   classifier.optLambda;
        %
        test_feats = curFeats(:,~isTrain & curValids);
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
% % % % save('~/storage/misc/baseLine_classifiers_5.mat','res');

%%
%% clc;
% A more concise summary
writeOutput = true;
writeImg = false;
tostd = 1;
ensuredir('~/notes/images/2014_06_23');
if writeOutput
    if tostd , fid = 1; else
        fid = fopen('~/notes/images/2014_06_23/res.txt','w');
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
    %     valids = cat(1,all_feats(feat_ids).isValid);
    for t = 1:length(classifiers)
        w = classifiers(t).w;
        all_hs(t,:) = w(1:end-1)'*all_feats(feat_ids(t)).feats;
    end
    %     all_hs(5,:) = 0;
    subsets = allSubsets(length(ress));
    
    %     subsets(:,5) = 0;
    %     subsets = [1 1 1 1];
    %     subsets = subsets(sum(subsets,2)<=2,:);
    weights = ones(length(all_feats),1);
    %     weights(2) = 5;
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
        [recall, precision, info] = vl_pr(2*(test_labels(valids(~isTrain))==iClass)-1, test_scores(valids(~isTrain)));
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