%%%%%%% Experiment 0044 %%%%%%%%%%%
%%%%%%% June 23, 2014 %%%%%%%%%%%%%%

%% The purpose of this experiment is to construct a baseline for differentiating
% between similar action classes, on the fra_db

%% initialization
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    
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
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV/'));
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
    featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
end

%%
%% extract sub images and corresp . features
%%
% feat_data_path = '~/storage/misc/fra_db_feats_new.mat';
% % if (exist(feat_data_path,'file'))
% %     load(feat_data_path);
% % else
%S

% for iNormalization = 1:length(normalizations)
iNormalization = 4;
all_feats = [];
% all_feats_2 = struct('feat',{},'type',{},'name',{},'isTrain',{},'srcImageIndex',{},...
%     'is_gt_location',{},'flipped',{},'isValid',{});
iFeat = 0;
ticId = ticStatus('feature extraction',1,.1);

roiParams.useCenterSquare = false;
curSuffix = 'regular';
if (roiParams.useCenterSquare)
    curSuffix = 'square';
end

for k = 1:1:length(fra_db)
    % check if this has been pre-computed.
    curFeatPath = j2m('~/storage/s40_fra_feats',fra_db(k).imageID);
    if (exist(curFeatPath,'file'))
        L = load(curFeatPath);
        if (roiParams.useCenterSquare)
            imgFeats = L.res.imgFeatsSquare;
        else
            imgFeats = L.res.imgFeats;
        end
        
        for tt = 1:length(imgFeats) % retain only the features with the chosen normalization
            if (~isempty(imgFeats(tt).feat))
                imgFeats(tt).feat = imgFeats(tt).feat(:,iNormalization);
            end
        end
        
        iFeat = iFeat + length(imgFeats);
        all_feats = [all_feats,imgFeats];
    else
        curImageData = fra_db(k);
        all_feats = [all_feats,fra_extract_feats_helper(conf,curImageData,roiParams,featureExtractor)];
    end
    tocStatus( ticId, k/length(fra_db));
end

fprintf('\n');
% %     save(feat_data_path,'all_feats_2','-v7.3');
% % end
% %
% get the object features for each test image.
% obj_feats = {};
% load ~/storage/misc/fra_class_priors.mat; % class_priors
% for useSquare = [false true]
% roiParams.useCenterSquare = false;
% curSuffix = 'regular';
% if (roiParams.useCenterSquare)
%     curSuffix = 'square';
% end

%% train on different feature types
% for 64
%classifierPath = ['~/storage/misc/baseLine_classifiers_5_' curSuffix '_' normalizations{iNormalization} num2str(size(featureExtractor.bowConf.bowmodel.vocab.means,2)) '_try.mat']
% for 1x2
classifierPath = ['~/storage/misc/baseLine_classifiers_5_' curSuffix '_' normalizations{iNormalization} num2str(size(featureExtractor.bowConf.bowmodel.vocab.means,2)) '_try1x2.mat']
% all other regions are negatives.
% classifierPath = ['~/storage/misc/baseLine_classifiers_5_' curSuffix '_' normalizations{iNormalization} num2str(size(featureExtractor.bowConf.bowmodel.vocab.means,2)) '_try1x2_negs.mat']
if (exist(classifierPath,'file'))
    load(classifierPath);
else
    
    %     res = struct('className',{},'class_id',{},'feat_id',{},'recall',{},'precision',{},'info',{},'classifier',{},...
    %         'feat_name',{});
    
    %     train_obj_feats = obj_feats(isTrain,:);
    all_feat_types = [all_feats.type];
    train_set = [all_feats.isTrain];
    [feat_types_u,ia] = unique(all_feat_types);
    feat_names = {all_feats(ia).name};
    all_image_inds = [all_feats.srcImageIndex];
    all_class_ids = class_labels(all_image_inds);
    is_gt_region = [all_feats.is_gt_location];
    
    valid_feats = [all_feats.isValid];
    n = 0;
    for iFeatType = 1:length(feat_types_u)
        curFeatType = feat_types_u(iFeatType)
        sel_feat_type = valid_feats & all_feat_types == curFeatType;
        for iClass = 1:length(classes)
            n = n+1;
            %             if (iFeatType~=3)
            %                 continue;
            %             end
            %             if (iClass ~=1)
            %                 continue;
            %             end
            res(n).className = classNames{iClass};
            res(n).class_id = iClass;
            sel_class = all_class_ids==iClass;
            sel_pos = train_set & sel_feat_type & sel_class;
            %sel_neg = train_set & sel_feat_type & ~sel_class;
            sel_neg = train_set & valid_feats & ~sel_class;
            
            features_pos = cat(2,all_feats(sel_pos).feat);
            features_neg = cat(2,all_feats(sel_neg).feat);
            
            % get more negative features
            % %             if(strcmp(feat_names{iFeatType},'obj'))
            % %                 moreNegFeats = {};
            % %                 sel_more_neg = isTrain & ([fra_db.classID]~=iClass);
            % %                 nImgs = 100;
            % %                 nNegatives = 20000;
            % %                 nBoxesPerImage = nNegatives/nImgs;
            % %                 img_sel = vl_colsubset(find(sel_more_neg),nImgs);
            % %                 for iNegImg = 1:length(img_sel)
            % %                     iNegImg
            % %                     imgData = fra_db(img_sel(iNegImg));
            % %                     [mcg_boxes,I] = get_mcg_boxes(conf,imgData,roiParams);
            % %                     imgFeats = load(j2m('~/storage/s40_fra_selective_search_feats',imgData));
            % %                     imgFeats = imgFeats.res;
            % %                     goods = find(~any(isnan(imgFeats)));
            % %                     curNegIndices = vl_colsubset(goods,nBoxesPerImage);
            % %                     moreNegFeats{iNegImg} = imgFeats(:,curNegIndices);
            % %                     %featureExtractor.extractFeatures(I,rois,'normalization',normalizations{iNormalization});
            % %                 end
            % %
            % %                 features_neg = [features_neg,cat(2,moreNegFeats{:})];
            % %                 %%
            % %                 %%
            % %             end
            
            classifier = train_classifier_pegasos(double(features_pos),double(features_neg),1);
            res(n).classifier.w =           classifier.w;
            res(n).classifier.optAvgPrec =  classifier.optAvgPrec;
            res(n).classifier.optLambda =   classifier.optLambda;
            res(n).feat_name = feat_names{iFeatType};
            res(n).feat_id = curFeatType;
            % calculate results for each image...
            sel_feat_type_test = sel_feat_type & ~train_set;
            features_test = cat(2,all_feats(sel_feat_type_test).feat);
            test_scores = res(n).classifier.w(1:end-1)'* features_test +res(n).classifier.w(end);
            test_inds = all_image_inds(sel_feat_type_test);
            final_scores = -inf(size(fra_db));
            for t = 1:length(test_scores)
                curInd = test_inds(t);
                final_scores(curInd) = max(final_scores(curInd),test_scores(t));
            end
            res(n).test_scores = final_scores;
            %             break;
        end
    end
    
    save(classifierPath,'res');
end

% % end
%% automatically found object features
load classInfo.mat
debugging = true;
objTypes = {'head','hand','obj','mouth'};
if (~debugging)
    obj_feats = cell(length(objTypes),length(fra_db),length(classes));
    gt_obj_feats = cell(length(fra_db),length(classes));
end
ticId = ticStatus('automatic object feature extraction',1,.1);
pk = 1:length(fra_db);

all_sums = {};
all_areas = {};
all_scores = {};

all_max_probs = zeros(length(fra_db),length(classes));

for ik = 1:length(fra_db)
    k = pk(ik);
    %     k
    curImageData = fra_db(k);
    if (curImageData.isTrain),continue,end;
    [rois,subRect,I,scaleFactor] = get_rois_fra(conf,curImageData,roiParams);
    
    max_probs = zeros(size(classes));
    
    for iClass =1:length(classes)
        probPath = fullfile('~/storage/s40_fra_box_pred_new',[curImageData.imageID '_' classNames{iClass} '_' objTypes{3} '.mat']);
        load(probPath);
        max_probs(iClass) = max(pMap(:));
    end
    all_max_probs(ik,:) = max_probs;
    scoresPath = j2m('~/storage/s40_fra_selective_search_classify',curImageData);
    scoresData = load(scoresPath);
    scores = max(scoresData.res.scores,scoresData.res.scores_flip);
    goods = find(~any(isnan(scores)));
    salMap = foregroundSaliency(conf,curImageData.imageID);
    salMap = imResample(cropper(salMap,subRect),size2(I));
    
    
    scores = scores(:,goods);
    
    all_classScores = zeros(1,5);
    
    % load the sums for the bounding box probs.
    L_prob_sums = load(j2m('~/storage/s40_fra_selective_search_sum_prob_maps',curImageData));
    areas = L_prob_sums.res.areas(goods);
    sums = L_prob_sums.res.sums(goods,:,3);
    
    for iClass =1:length(classes)
        
        sel_class = strmatch(classNames{iClass},{res.className});
        sel_feat = strmatch('obj',{res(sel_class).feat_name});
        cur_sel = sel_class(sel_feat);
        
        classScores = scores(cur_sel,:);
        
        all_sums{ik,iClass} = sums(:,iClass);
        all_areas{ik,iClass} = areas(:);
        all_scores{ik,iClass} = classScores(:);
        
    end
    tocStatus( ticId, k/length(fra_db));
end

%%
class_scores = -inf(length(fra_db),length(classes));
all_max_probs2 = all_max_probs./repmat(sum(all_max_probs,2),1,size(all_max_probs,2));
for ik = 1:length(fra_db)
    k = pk(ik);
    curImageData = fra_db(k);
    
    if (curImageData.isTrain),continue,end;
    all_classScores = zeros(1,5);
    for iClass =1:length(classes)
        
        curScores = all_scores{ik,iClass};
        if (isempty(curScores))
            continue;
        end
        curSums = all_sums{ik,iClass}./all_areas{ik,iClass};        
        [s,smin,smax] = normalise(curScores);
        scores_normalized = s(:).*curSums;
        scores_normalized = denormalise(scores_normalized,smin,smax);
        %         scores_normalized(all_priors{k,iClass} = scores_normalized(all_priors{k,iClass}<.1)-.1;
        class_scores(ik,iClass) = max(scores_normalized);
                
%         scores1 = curSums;%+0all_priors{k,iClass};
%         class_scores(ik,iClass) = max(scores1);
%         class_scores(ik,iClass) = all_max_probs2(ik,iClass);
        
        %                 AA = computeHeatMap(I,[mcg_boxes,scores_normalized],'max');
        %                 clf,imagesc2(sc(cat(3,AA,I),'prob'));
        %                 disp([classNames{iClass} ' ' num2str(max(AA(:)))]);
        %                 pause;
    end
end

% class_scores = class_scores./repmat(sum(class_scores,2),1,size(class_scores,2));

%
% save('~/storage/misc/fra_db_with_obj_feats.mat', 'obj_feats', '-v7.3');
% get the maximal score for each image
load curScores;
probScores = curScores;
% probScores = all_max_probs;
% probScores = probScores./repmat(sum(probScores,2),1,size(probScores,2));
% probScores = probScores./repmat(max(probScores,[],2),1,size(probScores,2));

% now, replace the final scores of the objects by those obtained when finding
% the object region automatically.
manual_mode_names = {'all','none','class_only','non_class_only','prob_scores'};
isTest = ~isTrain;
for iClass = 1:length(classes)
    sel_class = strmatch(classNames{iClass},{res.className});
    sel_feat = strmatch('obj',{res(sel_class).feat_name});
    cur_sel = sel_class(sel_feat);
    auto_scores = class_scores(:,iClass);
    auto_scores(isTrain) = -inf;
    auto_scores = row(auto_scores);
    scores = repmat(auto_scores,5,1);
    isClass = [fra_db.classID]==iClass;
    % first mode: all scores are manual
    scores(1,:) = res(cur_sel).test_scores;
    % second mode: all are automatic (nothing to be done)
    % 3rd mode: only class objects are manually given
    scores(3,isClass) = res(cur_sel).test_scores(isClass);
    % 4th mode: only non-class objects are manually given
    scores(4,~isClass) = res(cur_sel).test_scores(~isClass);
    scores(5,~isTrain) = probScores(~isTrain,iClass);
    scores(5,:) = scores(5,:) + 0*scores(2,:);
    res(cur_sel).test_scores_auto = scores;
    %                 res(cur_sel).test_scores_auto(:,isinf(res(cur_sel).test_scores)) = -inf;
end
%% a concise summary
figure
test_object_detection = false;
manual_mode_sel = [1 5];

% A concise summary
writeOutput = true;
writeImg = false;
tostd = 1;
showDir = fullfile('~/notes/images/',datestr(now,'yyyy_mm_dd'));
ensuredir(showDir);
if writeOutput
    if tostd , fid = 1; else
        fid = fopen(fullfile(showDir,'res.txt'),'w');
    end
end
lineStyles = {'-.','--','-'};
markerStyles = {'s','o','d','*'};
feat_abbr = {res.feat_name};
res_class_names = {res.className};
scores_all_methods = {};
% validImages =
%
% cur_class_sel = [fra_db.classID]==5;
% hist([fra_db(~isTrain).classID]);
% figure,hist([fra_db(~isTrain &validImages).classID]);

nFeat = length(feat_names);
pdfPaths = {};

if (test_object_detection)
    mode_names = manual_mode_names(manual_mode_sel);
else
    mode_names = {'automatic'};
end
sss = zeros(1,4);
maximalCombination = 5;
minimalCombination = 1;

summary = struct('iClass',{},'className',{},'subset',{},'square_feats',{},'imanual_mode',{},...
    'mode_name',{},'recall',{},'precision',{},'info',{},'description',{});
iSummary = 0;

for iClass = 1:length(classNames)
    for iManualMode = 1:length(mode_names)
        %         iClass
        ret_fig = clf;hold on;
        set(gcf,'DefaultAxesLineStyleOrder',lineStyles)
        curClassName = classNames{iClass};
        
        sel_class = strmatch(classNames{iClass},{res.className});
        ress = res(sel_class);
        if (test_object_detection)
            sel_obj_feat = strmatch('obj',{ress.feat_name});
            ress(sel_obj_feat).test_scores = ress(sel_obj_feat).test_scores_auto(manual_mode_sel(iManualMode),:);
        end
        
        feat_ids = [ress.feat_id];
        curFeatNames = {ress.feat_name};
        all_hs = cat(1,ress.test_scores);
        % % %         all_hs(:,~validImages) = -inf; %TODO!
        subsets = allSubsets(length(ress));
        subsets = subsets(sum(subsets,2)<=maximalCombination,:);
        subsets = subsets(sum(subsets,2)>=minimalCombination,:);
        colors = hsv(size(subsets,1));
        subsets = [1 1 1 1 1];
        subsets = [0 0 0 1 0];
        weights = ones(length(ress),1);
        method_s = {};
        ap_s = {};
        mStrings = {};
        nColor = 1:64;
        for iChoice = 1:size(subsets,1)
            choice = subsets(iChoice,:)';
            curWeights = weights.*choice;
            scores_total = curWeights'*all_hs;
            scores_all_methods{end+1} = scores_total;
            tChoice = nColor(iChoice);
            sel_ = ~isTrain & class_labels ~=0;
            test_labels = class_labels(sel_);
            test_scores = scores_total(sel_);
            %                         assert(none(isnan(test_scores)));
            test_scores(isnan(test_scores)) = -inf;
            test_scores = test_scores + rand(size(test_scores))*.001;
            [recall, precision, info] = vl_pr(2*(test_labels==iClass)-1, test_scores,'IncludeInf',false);
            plot(recall,precision,'color',colors(tChoice,:), 'LineWidth',2);
            curMethods = curFeatNames(choice>0);
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
            iSummary = iSummary+1;
            summary(iSummary).subset = choice;
            summary(iSummary).iClass = iClass;
            summary(iSummary).className = curClassName;
            summary(iSummary).mode_name = mode_names{iManualMode};
            summary(iSummary).square_feats = roiParams.useCenterSquare;
            summary(iSummary).recall = recall;
            summary(iSummary).precision = precision;
            summary(iSummary).info = info;
            summary(iSummary).ap = info.ap;
            summary(iSummary).description = mString;
            summary(iSummary).imanual_mode = iManualMode;
            
        end
        xlabel('recall'); ylabel('precision');
        h = get(gca,'Title');
        curTitle=get(h,'String');
        set(h,'interpreter','none');
        %     h = get(gcf,'Legend');
        %     set(gca,);
        lh = legend(mStrings,'interpreter','none');
        if (~test_object_detection)
            suff = '';
        else
            suff = ['-manual:' mode_names{iManualMode}];
        end
        
        title([curClassName suff]);
        aps = cat(1,ap_s{:}); %[r,ir] = sort(aps,'descend');
        %     maximizeFigure;
        [r,ir] = sort(aps,'descend');
        ir = ir(1:min(3,length(ir)));
        ch = get(lh,'Children');
        th = findobj(ch,'Type','text');
        headerhand = findobj(th,'String',mStrings{ir(1)});
        for i = 1:length(th)
            if (th(i) == headerhand)
                rmappdata(th(i),'Listeners');
            end
        end
        set(headerhand,'FontWeight','bold','FontSize',12);
        pad_size = 10;
        padel_string = sprintf('%%-%ds',pad_size);
        padel_f = sprintf('%%-%d.3f',pad_size);
        if (writeOutput)
            fprintf(fid,'\n%s:\n',[curClassName suff]);
            for t = 1:nFeat
                fprintf(fid,padel_string,feat_names{t});
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
            
            % find the mean ranking of each feature
            %             ss = sum(subsets(ir(1:round(end/2)),:))./sum(subsets(ir(round(end/2)+1),:));
            %             sss = sss+ss;
            if (writeImg)
                pdfPath = fullfile(showDir,[curClassName suff '.pdf']);
                save2pdf(pdfPath);
                pdfPaths{end+1} = pdfPath;
                set(gcf,'units','normalized','outerposition',[0 0 1 1]);
                %export_fig(fullfile(showDir,[curClassName '.pdf']));
            end
        end
        if (tostd)
%                         pause;
        end
        %         pause
    end
end

lineStyles = linspecer(2);
infos = [summary.info];
manuals = [summary.imanual_mode]==1;
manual_ap = col([infos(manuals).ap]);
autos = [summary.imanual_mode]==2;
auto_ap = col([infos(autos).ap]);

figure,
bar([manual_ap auto_ap]); colormap(lineStyles);

set(gca,'XTickLabel',classNames);
ylabel( 'Avg. Precision');
legend({'manually specified objects','automatically detected objects'});
