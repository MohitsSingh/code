function [w b info]  = trainRegionDetector(conf,roiPath)
% trains a region-based detector.
info = [];

% get regions of interest for action.
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[action_rois,true_ids] = markActionROI(conf,roiPath);
false_ids = train_ids(~train_labels);

load('~/code/kmeans_4000.mat','codebook');
model.numSpatialX = [2];
model.numSpatialY = [2];
model.vocab = codebook;
model.w = [];
model.b = [];

% get features from all positive boxes
pos_sel = 1:length(true_ids);
if (exist('posFeats.mat','file'))
    load('posFeats.mat');
else
    posFeats = getPosFeats(conf,true_ids(pos_sel),action_rois(pos_sel,:),model);
    posFeats = posFeats(:,~isnan(sum(posFeats)));
    save('posFeats.mat','posFeats');
end

% posFeats = posFeats(:,[2 5 9 14 20 21 23 28 29 41 44 45]);

% get features from a random subset of negative boxes.
% % % if (exist('negFeats_raw.mat','file'))
% % %     load('negFeats_raw.mat');
% % % else
% % %     negFeats = getNegFeats(conf,false_ids,model);
% % %     negFeats = negFeats(:,~isnan(sum(negFeats)));
% % %     save('negFeats.mat','negFeats');
% % % end

% posFeats = sparse(double(posFeats));
% posFeats = vl_homkermap(posFeats, 1, 'kchi2', 'gamma', .7);
% posFeats = sparse(double(posFeats));
% negFeats = vl_homkermap(negFeats, 1, 'kchi2', 'gamma', .7);
%negFeats = vl_colsubset(negFeats,5000);
matObj = matfile('negFeats_raw.mat');
% posFeats = sparse(double(posFeats));
negFeats = [];
for z = 1:1000:1001
    z
    negFeats = [negFeats,(double(matObj.negFeats(:,z:z+999)))];    
end
% z = 1;
% negFeats = sparse(double(matObj.negFeats(:,z:z+99)));
% [w,b] = train_classifier(posFeats,negFeats,.001,size(negFeats,2)/size(posFeats,2));
[w b info] = trainClassifier(conf,posFeats,negFeats);
% extract features from positive images.
    function posFeats = getPosFeats(conf,true_ids,action_rois,model)
        
        %TODO - try two modes, one will be choosing as positive samples
        % the regions which correspond to the segments best overlapping the
        % ground-truth bounding boxes, and the other will be the bounding boxes
        % themselves.
        % You can also add the shape of the segments as an explicit cue.
        conf.not_crop = true;
        masks = {};
        feats = struct('frames',[],'descrs','binsa');
        for k = 1:length(true_ids)
            k
            [fullImage,xmin,xmax,ymin,ymax] = getImage(conf,true_ids{k});
            bowFile = fullfile(conf.bowDir,strrep(true_ids{k},'.jpg','.mat'));
            load(bowFile,'F','bins');
            masks{k} = drawBoxes(fullImage,action_rois(k,:),1,1)>0;
%             clf;imshow(bsxfun(@times,masks{k},fullImage));
%             pause;
            feats(k).frames = F;
            feats(k).binsa = row(bins);
            feats(k).descrs = [];
        end
        
        posFeats = getBOWFeatures(conf,model,true_ids,masks,feats);
    end

    function negFeats = getNegFeats(conf,false_ids,model)
        conf.not_crop = true;
        
        negFeats = {};        
        maxRegions = 15000;
        masks = {};
        imageSubset = vl_colsubset(row(false_ids),1000);
        regionsPerImage = round(maxRegions/length(imageSubset));
        for k = 1:length(imageSubset)
            k
            %             [fullImage,xmin,xmax,ymin,ymax] = getImage(conf,imageSubset{k});
            bowFile = fullfile(conf.bowDir,strrep(imageSubset{k},'.jpg','.mat'));
            regionFile = fullfile(conf.gpbDir,strrep(imageSubset{k},'.jpg','_regions.mat'));
            load(bowFile,'F','bins');
            load(regionFile,'regions','regionOvp');
            
            % remove regions whos are is too small...
            regions = row(regions);
            minArea = 1000;
            validArea = cellfun(@(x)sum(x(:)),regions) > minArea;                                    
            regions = regions(validArea);
            regionOvp = regionOvp(:,validArea);
            regionOvp = regionOvp(validArea,:);
            %regions = vl_colsubset(regions,regionsPerImage);            
            %p = randperm(length(regions));
            p = length(regions):-1:1;
            visited = false(size(regions));
            iRegion = 1;
            regionCount = 0;
            numSkipped = 0;
            while (regionCount < regionsPerImage && iRegion <= length(regions))
                curCandidate = p(iRegion);
                if (iRegion == 1)
                    visited(p(iRegion)) = true;
                    regionCount = regionCount + 1;
                else
                    ovps = max(regionOvp(curCandidate,visited));
                    if (max(ovps) <= .5)
                        visited(p(iRegion)) = true;
                        regionCount = regionCount + 1;
                    else
                        numSkipped = numSkipped + 1;
                    end  
                end
                iRegion = iRegion + 1;
            end
            numSkipped
            regions = regions(visited);
            
%             for c = 1:length(regions)
%                 imagesc(regions{c});
%                 pause(.1);
%             end
%             
            % select non overlapping regions.
%             figure,imagesc(region)
            
            for iRegion = 1:length(regions) % eliminate boundaries between sub-regions
                regions{iRegion} = imclose(regions{iRegion},ones(3));
            end
            masks = regions;
            feats = struct('frames',[],'descrs','binsa');
            feats.frames = F;
            feats.binsa = row(bins);
            feats.descrs = [];
            negFeats{k} = getBOWFeatures(conf,model,imageSubset(k),{masks},feats);
        end
        
        negFeats = cat(2,negFeats{:});
    end
end