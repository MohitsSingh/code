function [regionData] = collectRegionsAndFeatures(conf,imgData,use_gt_regions,nPerImage,featureExtractor)
regionData = struct('imgIndex',{},'region',{},'feats',{});
mcgDir = '~/storage/fra_face_seg';
segFeatsDir = '~/storage/fra_db_face_seg_feats';

%regionFeatsDir = '~/storage/fra_db_face_region_dnn_feats';
% regionFeatsDir = '~/storage/fra_db_face_region_dnn_feats';
tic_id = ticStatus('extracting region features...',.5,.5,true);
valids = false(size(imgData));
% for t = 1:length(imgData)
%     valids(t) = false;
% end

regionData = struct('imgIndex',{},'region',{},...
                'feats',{},'valid',{});

for t = 1:length(imgData)
    %     t
    if ~all(imgData(t).face_landmarks.valids)
        regionData(t).valid = false;
        continue
    end
    curImgData = imgData(t);
    I = curImgData.I;
    sz =size2(I);
    if use_gt_regions        
%         curRegion = box2Region(inflatebbox( mean(curImgData.gt_obj),size2(I)/2,'both',true),size2(I));        
        curRegion = poly2mask2(curImgData.gt_obj,size2(I));
        regionData(t).region = curRegion;
        regionData(t).imgIndex = t;        
        %regionData(t).feats = featureExtractor.extractFeatures(I,curRegion);
        regionData(t).feats = myExtractFeatures(curImgData,curRegion,featureExtractor);
        regionData(t).valids = true;
    else
        % load segmentation
        
        doBoxes = false;
        if (~doBoxes)            
            load(j2m(mcgDir,curImgData));
            % choose subset of segments
            candidates = res.candidates;
            sel_ = removeBadRegions(candidates,I);
            nPerImage1 = min(nPerImage,length(sel_));
            si = weightedSample(1:length(sel_), candidates.scores(sel_), nPerImage1);
            sel_ = sel_(si);
            nRegions = length(sel_);                        
            candidates.labels = candidates.labels(sel_);
            candidates.cand_labels = candidates.cand_labels(sel_,:);
            candidates.bboxes = candidates.bboxes(sel_,:);
            candidates.scores = candidates.scores(sel_);            
            %regions = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);                        
            if (1)                                
                regions = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);                
                %         regions2 = my_cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);                
                nRegions = size(regions,3);
                r = {};
                for u = 1:nRegions
                    r{u} = regions(:,:,u);
                end
            else
                r = {};
                for u = 1:nRegions
                    r{end+1} = box2Region(candidates.bboxes(u,:),sz);
                end
            end
            
        else
            r = {};
            bb = makeTiles(I,4^2,4);
            ss = vl_colsubset(1:size(bb,1),nPerImage,'random');
            bb = bb(ss,:);
            nRegions = size(bb,1);
            
            for u = 1:nRegions
                r{end+1} = box2Region(bb(u,:),sz);
            end
        end
        curFeats = myExtractFeatures(curImgData,r,featureExtractor);
%         featureExtractor.extractFeatures(I,r);
        for z = 1:nRegions
            regionData(end+1) = struct('imgIndex',t,'region',r{z},...
                'feats',curFeats(:,z),'valid',true);            
        end
        
    end
    tocStatus(tic_id, t/length(imgData));
end


function rr = my_cands2masks(cand_labels, f_lp, f_ms)
n = size(cand_labels,1);
rr = false([size(f_lp) n]);
for t = 1:size(cand_labels,1)
    rr(:,:,t) = ismember(f_lp,cand_labels(t,:));
end


%function goods = removeBadRegions(regions,I)
function sel = removeBadRegions(candidates,I)
bb = candidates.bboxes;
[~,~,areas] = BoxSize(bb);
a = prod(size2(I));
sel = find(areas/a < .5);


function feats = myExtractFeatures(imgData,regions,featureExtractor)
    if ~iscell(regions)
        regions = {regions};
    end
    I = imgData.I;
    appearanceFeats = featureExtractor.extractFeatures(I,regions);
    occupancyFeats = cellfun2(@(x) col(imResample(single(x),[5,5],'bilinear')), regions);
    occupancyFeats = cat(2,occupancyFeats{:});
    
    % min,max distance to all facial features
%     z = {};
    xy = imgData.face_landmarks.xy;
    mindist_feats = {};
    
    for t = 1:length(xy)
        zz = zeros(size2(I));
        zz(round(xy(t,2)),round(xy(t,1))) = 1;
        z = bwdist(zz);
        mindist_feats{t} = row(cellfun(@(x) min(z(x)),regions));
    end
    
    mindist_feats = cat(1,mindist_feats{:});
    
    feats = [appearanceFeats;occupancyFeats;mindist_feats];
    
    
    

    