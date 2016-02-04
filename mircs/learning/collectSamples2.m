function sampleData = collectSamples2(conf, fra_db,cur_set,params,debug_stuff)
dlib_landmark_split;
if params.testMode
    tic_id = ticStatus('extracting test features...',.5,.5,false);
else
    tic_id = ticStatus('extracting training features...',.5,.5,false);
end
samples = {};
if nargin < 5
    hasDebug = false;
else
    hasDebug = true;
end
debug_jump = params.debug_jump;
phases = params.phases;
for iImage = 1:debug_jump:length(cur_set)
    % for iImage = 1
    
    %     iImage
    imgInd = cur_set(iImage);
    imgData = fra_db(imgInd);
    testFeatsPath = j2m('~/storage/testFeats',imgData);
    if params.testMode
        if exist(testFeatsPath,'file')
            load(testFeatsPath)
            samples{end+1} = curSamples;            
        end
    else
        I = getImage(conf,imgData);
        [I_sub,~,mouthBox] = getSubImage2(conf,imgData,~params.testMode);
        [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,imgData.isTrain);
        [groundTruth,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox);
        %     if ~isValid && ~params.testMode
        %         warnString = sprintf('invalid image found : %d --> %s\nskipping this image!!!',...
        %             imgInd,imgData.imageID);
        %         params.logger.warn(warnString,'collectSamples');
        %     else
        if ~isValid
            warnString = sprintf('empty ground truth found : %d --> %s\nsetting this ground truth to empty image',...
                imgInd,imgData.imageID);
            groundTruth = false(size2(I_sub));
            params.logger.warn(warnString,'collectSamples');
        end
        % coarse stage:
        %         if params.testMode
        %             imgData.action_obj.active = imgData.action_obj.test;
        %         else
        %             imgData.action_obj.active = imgData.action_obj.train;
        %         end
        imgData.I = I;
        imgData.I_sub = I_sub;
        imgData.mouthMask = mouthMask;
        imgData.curLandmarks = curLandmarks;
        imgData.mouthBox = mouthBox;
        % Stage 1: get candidates using coarse prediction of convnet
        iPhase=1;
        curPhase = phases(iPhase);
        P = curPhase.alg_phase;
        regions = P.getCandidates(imgData);
        if isempty(regions)
            errorString= sprintf('no regions found for image: %d --> %s\n',...
                imgInd,imgData.imageID);
            params.logger.error(errorString,'collectSamples');
            labels = [];
            ovps = [];
            curSamples.feats = {};
        else
            % sample (only does something in case of training)
            [regions,labels,ovps] = P.sampleRegions(regions,...
                groundTruth);
            
            if (hasDebug && debug_stuff.calcFeats) || ~hasDebug
                interaction_features = phases(2).alg_phase.extractFeatures(imgData,regions);
                app_and_shape = phases(3).alg_phase.extractFeatures(imgData,regions);
                %             appearanceFeatures = 0;%phases(3).alg_phase.extractFeatures(imgData,regions);
                curSamples.feats = [interaction_features,app_and_shape];
            else
                curSamples.feats = {};
                
            end
            
            %             curSamples.imgInd = imgInd;
            %             curSamples.imgInd_s = iImage;
            %             curSamples.regions = regions;
            %             curSamples.labels = labels;
            %             curSamples.I_sub = I_sub;
            %             %curSamples.feats = currentFeats;
            %             curSamples.ovps = ovps;
            %             samples{end+1} = curSamples;
        end
        
        curSamples.imgInd = imgInd;
        curSamples.imgInd_s = iImage;
        curSamples.regions = regions;
        curSamples.labels = labels;
        curSamples.imageID = imgData.imageID;
        curSamples.I_sub = I_sub;
        %curSamples.feats = currentFeats;
        curSamples.ovps = ovps;
        if params.testMode && ~exist(testFeatsPath,'file')
            save(testFeatsPath,'curSamples');
        end
        
        samples{end+1} = curSamples;
    end
    tocStatus(tic_id,iImage/length(cur_set));
    %     end
end
sampleData = aggregateSamples(fra_db,samples);
%function [feats,labels,regions,inds] = aggregateSamples(fra_db, samples)
function sampleData = aggregateSamples(fra_db, samples)
sampleData = struct('inds',{},'feats',{},'labels',{},'regions',{});
labels = {};
feats = {};
inds = {};
inds_s = {};
regions = {};
imgs = {};
for t = 1:length(samples)
    curSamples = samples{t};
    imgInd = curSamples.imgInd;
    imgInd_s = curSamples.imgInd_s;
    curClassLabel = fra_db(imgInd).classID;
    curLabels = curSamples.labels;
    curOvps = curSamples.ovps;
    curLabels(curLabels==1) = curClassLabel;
    inds{t} = ones(1,length(curLabels))*imgInd;
    inds_s{t} = ones(1,length(curLabels))*imgInd_s;
    labels{t} = curLabels(:);
    ovps{t} = curOvps(:);
    regions{t} = row(curSamples.regions);
    imgs{t} = curSamples.I_sub;
    %feats{t} = [curSamples.f_int;curSamples.f_app;curSamples.f_shape]
    feats{t} = curSamples.feats;
end
inds = cat(2,inds{:});
inds_s = cat(2,inds_s{:});
feats = cat(1,feats{:});
labels = cat(1,labels{:});
ovps = cat(1,ovps{:});
regions = cat(2,regions{:});
sampleData(1).inds = inds;
sampleData.inds_s = inds_s;
sampleData.feats = feats;
sampleData.labels = labels;
sampleData.ovps = ovps;
sampleData.regions = regions;
sampleData.imgs = imgs;
sampleData.imageIDS = cellfun2(@(x) x.imageID,samples);