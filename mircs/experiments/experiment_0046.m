%% experiment 0046
%% create a generic prior of the location of the objects given other regions.
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
    featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
    featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
    initialized = true;
end
res = struct('sal',{},'sal_bd',{},'bbox',{},'resizeRatio',{});
opts.show = false;
maxImageSize = 300;
opts.maxImageSize = maxImageSize;
spSize = 40;
opts.pixNumInSP = spSize;
conf.get_full_image = true;
data = struct('faceBox',{},'upperBodies',{},'isvalid',{},'subs',{});

% collect the relative location of the hand, head, object.
% sample a single object per object type from each image.
class_priors = {};

object_locations = struct('classID',{},'locs',{});
nObjectTypes = 4;
for iClass = 1:length(classes)
    f = find(class_labels==iClass & isTrain);
    z_class = zeros(roiParams.absScale);
    curLocs = nan(length(f),nObjectTypes,2);
    for ii = 1:length(f)
        ii
        curImageData = fra_db(f(ii));
        [rois,subRect,I] = get_rois_fra(conf,curImageData,roiParams);
        %
        for iObjType = 1:nObjectTypes
            obj_rois = rois([rois.id]==iObjType);
            if (~isempty(obj_rois))
                obj_rois = obj_rois(1);
                curLocs(ii,iObjType,:) = boxCenters(obj_rois.bbox);
            end
        end
        
        %         figure,imagesc2(I);
        %         plotPolygons(reshape(curLocs(ii,:),[],2),'r+');
        %         drawnow;pause;continue;
        
        obj_rois = rois([rois.id]==3);
        for iRoi = 1:length(obj_rois)
            z_class = z_class+poly2mask2(obj_rois(iRoi).poly,size2(z_class));
        end
    end
    
    object_locations(iClass).classID = iClass;
    object_locations(iClass).locs = curLocs;
    %     obj_rois = rois([rois.id]==3);
    %     for iRoi = 1:length(obj_rois)
    %         z_class = z_class+poly2mask2(obj_rois(iRoi).poly,size2(z_class));
    %     end
    % %
    z_class = normalise(z_class+fliplr(z_class));
    class_priors{iClass} = normalise(z_class);
end

% difference between: head-hand,hand-obj,head-obj

% figure,imagesc2(imResample(class_priors{3},[50 50]));
save '~/storage/misc/object_locations.mat' object_locations class_priors;

% get the distribution of distances between face-object and hand-object.
iHead = 1;iHand = 2;iObj = 3;iMouth = 4;

% 1-d distributions

diff_sets = [iHand iObj;iMouth iObj;iHead iObj];

D = [0:1:600 inf];
dist_stats = struct('diff_set',{},'dists',{},'prob',{});
for iClass = 1:length(classes)
    % first, hand-object.
    for iDiff =1:size(diff_sets,1)
        cur_diff_set = diff_sets(iDiff,:);
        curObjLocations = object_locations(iClass).locs(:,cur_diff_set,:);
        d = squeeze(curObjLocations(:,2,:)-curObjLocations(:,1,:));
        [ii,jj] = find(isnan(d));
        d(unique(ii),:) = [];
        dists = sum(d.^2,2).^.5;
        [N,bins] = histc(dists,D);
        dist_stats(iClass,iDiff).diff_set = cur_diff_set;
        dist_stats(iClass,iDiff).dists = D;
        dist_stats(iClass,iDiff).prob = cumsum(N)/sum(N);
    end
end
%     figure,plot(D,cumsum(N)/sum(N))
%%
smooth_class_priors = cellfun2(@(x) imfilter(x,fspecial('gauss',200,25),'same'),class_priors);
%%

for iClass = 1:length(classes)
    f = find(class_labels==iClass & isTest);
    for ii = 1:length(f)
        ii
        curImageData = fra_db(f(ii));
        [rois,subRect,I] = get_rois_fra(conf,curImageData,roiParams);
        handRoi = find([rois.id] == iHand,1,'first');
        objRoi = find([rois.id] == iObj,1,'first');
        mouthRoi = find([rois.id] == iMouth,1,'first');
        if (any(handRoi) && any(objRoi) && any(mouthRoi))
            handCenter = boxCenters(rois(handRoi).bbox);
            mouthCenter = boxCenters(rois(mouthRoi).bbox);
            objCenter = boxCenters(rois(objRoi).bbox);
        else
            continue;
        end
                
        %class_priors{3},[50 50])
        
        T = imResample(smooth_class_priors{iClass},size2(I));
        clf; imagesc2(sc(cat(3,T,I),'prob'));
        pause;continue;
        
        
        [xx,yy] = meshgrid(1:size(I,2),1:size(I,1));
        d_hand = ((xx-handCenter(1)).^2+(yy-handCenter(2)).^2).^.5;
        d_head = ((xx-mouthCenter(1)).^2+(yy-mouthCenter(2)).^2).^.5;
        
        % quantize distance map
        [~,b_bin] = histc(d_hand,D);
        b_bin_hand = 1-dist_stats(iClass,1).prob(b_bin);
        figure(2),clf,subplot(1,3,1);imagesc2(sc(cat(3,b_bin_hand,I),'prob')); title('hand distance')
        [~,b_bin] = histc(d_head,D);
        b_bin_mouth = 1-dist_stats(iClass,2).prob(b_bin);
        subplot(1,3,2),imagesc2(sc(cat(3,b_bin_mouth,I),'prob')); title('mouth distance');
        
        b_total = b_bin_hand.*b_bin_mouth;
        subplot(1,3,3),imagesc2(sc(cat(3,b_total,I),'prob')); title('combined distance');
        
        pause;continue
        
        
        DD = l2([d_hand(:) d_head(:)],X);
        DD = reshape(DD,[size2(I) size(DD,2)]);
        S = sum(exp(-DD/1000),3);
        figure(2),clf; imagesc2(sc(cat(3,S,I),'prob'));
        pause;
        %figure,imagesc2(sum(exp(-DD/1000),3));
        %         [r,ir] = min(DD,[],2);
        
        %         rr = reshape(r,size2(I));
        %         figure,imagesc2(exp(-rr/100));
        %         figure,imagesc2(I)
        
    end
    
end



diff_sets = [iHand iObj;iHead iObj;iMouth iObj];

D = 1:3:300;
dist_stats = struct('diff_set',{},'dists',{},'prob',{});
for iClass = 1:length(classes)
    % first, hand-object.
    for iDiff =1:size(diff_sets,1)
        cur_diff_set = [iHand iObj];
        curObjLocations = object_locations(iClass).locs(:,cur_diff_set,:);
        d = squeeze(curObjLocations(:,2,:)-curObjLocations(:,1,:));
        dists = l2(d,[0 0]).^.5;
        [N,bins] = hist(dists,D);
        dist_stats(iClass,iDiff).diff_set = cur_diff_set;
        dist_stats(iClass,iDiff).dists = D;
        dist_stats(iClass,iDiff).prob = cumsum(N)/sum(N);
        %     figure,plot(b,cumsum(a)/sum(a))
        %     figure,plotPolygons(l2(d,[0 0]),'r+');
        % save '~/storage/misc/fra_class_priors.mat' class_priors;
        
        %% common distribution: distance of object from hand and mouth
        locs = object_locations(iClass).locs;
        d1 = squeeze(locs(:,iObj,:)-locs(:,iMouth,:));
        d2 = squeeze(locs(:,iObj,:)-locs(:,iHand,:)); % you have a bug or bad data... check it!
        d1 = l2(d1,[0 0]).^.5;
        d2 = l2(d2,[0 0]).^.5;
        
        X = [d1 d2];
        [ii,jj] = find(isnan(X));
        X(unique(ii),:) = [];
        % X(ii,:) = [];
        figure,plot(d1,d2,'r+');
        %%
        for iClass = 2:length(classes)
            f = find(class_labels==iClass & isTrain);
            z_class = zeros(roiParams.absScale);
            curLocs = nan(length(f),nObjectTypes,2);
            for ii = 1:length(f)
                ii
                curImageData = fra_db(f(ii));
                [rois,subRect,I] = get_rois_fra(conf,curImageData,roiParams);
                handRoi = find([rois.id] == iHand,1,'first');
                objRoi = find([rois.id] == iObj,1,'first');
                mouthRoi= find([rois.id] == iHead,1,'first');
                if (any(handRoi) && any(objRoi) && any(mouthRoi))
                    handCenter = boxCenters(rois(handRoi).bbox);
                    mouthCenter = boxCenters(rois(mouthRoi).bbox);
                    objCenter = boxCenters(rois(objRoi).bbox);
                end
                [xx,yy] = meshgrid(1:size(I,2),1:size(I,1));
                d_hand = ((xx-handCenter(1)).^2+(yy-handCenter(2)).^2).^.5;
                d_head = ((xx-mouthCenter(1)).^2+(yy-mouthCenter(2)).^2).^.5;
                
                DD = l2([d_hand(:) d_head(:)],X);
                DD = reshape(DD,[size2(I) size(DD,2)]);
                S = sum(exp(-DD/1000),3);
                clf; imagesc2(sc(cat(3,S,I),'prob'));
                pause;
                %figure,imagesc2(sum(exp(-DD/1000),3));
                %         [r,ir] = min(DD,[],2);
                
                %         rr = reshape(r,size2(I));
                %         figure,imagesc2(exp(-rr/100));
                %         figure,imagesc2(I)
                
            end
            
        end
    end
end