function [regionData] = collectRegionsAndFeatures(conf,imgData,use_gt_regions,nPerImage,featureExtractor)
regionData = struct('imgIndex',{},'region',{},'feats',{});
mcgDir = '~/storage/fra_face_seg';
segFeatsDir = '~/storage/fra_db_face_seg_feats';

%regionFeatsDir = '~/storage/fra_db_face_region_dnn_feats';
% regionFeatsDir = '~/storage/fra_db_face_region_dnn_feats';
tic_id = ticStatus('extracting region features...',.5,.5,true);
for t = 1:length(imgData)
%     t
    curImgData = imgData(t);
    I = curImgData.I;
    if use_gt_regions
        curRegion = poly2mask2(curImgData.gt_obj,size2(I));
        regionData(t).region = curRegion;
        regionData(t).imgIndex = t;
        regionData(t).feats = featureExtractor.extractFeatures(I,curRegion);
    else
        
        load(j2m(mcgDir,curImgData));
        % load the features...
        regions = cands2masks(res.candidates.cand_labels, res.candidates.f_lp, res.candidates.f_ms);
        [goods] = removeBadRegions(regions,I);
        regions = regions(:,:,goods);        
        nRegions = size(regions,3);
        nPerImage1 = min(nPerImage,nRegions);        
        si = weightedSample(1:nRegions, res.candidates.scores(goods), nPerImage1);
        r = {};
        for u = 1:length(si)
            r{u} = regions(:,:,si(u));
        end
        load(j2m(segFeatsDir,curImgData));
        curFeats = feats(:,goods);
        curFeats = feats(:,si);
        %featureExtractor.extractFeatures(I,r);
        for z = 1:length(si)           
            regionData(end+1) = struct('imgIndex',t,'region',r{z},...
                'feats',curFeats(:,z));
        end
    end
    tocStatus(tic_id, t/length(imgData));
end

function goods = removeBadRegions(regions,I)
areas = squeeze(sum(sum(regions,1),2));
a = prod(size2(I));
goods = areas/a < .5;
% end