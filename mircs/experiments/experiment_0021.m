%%%% Experiment 21 %%%%
%% Jan 15/2014
% The purpose of this experiment is to use the fast no-hard-negative-mining
% code in order to learn several object categories.
% In order to do so, I first will collect some negative images from which
% all the negatives will be collected. Then I shall convert my ground-truth
% structure into something which can be processed by the circulant code.
% finally I'll do a bit of testing.
if (0)
            
    initpath;
    config;
    init_hog_detectors;
    %     init_hog_detectors_sqr;
    refine_hog_detectors;
    load ~/storage/misc/imageData_new;
    % now we have weight vectors for all of the above...
    addpath('/home/amirro/code/3rdparty/PolygonClipper');
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
        
    %% occlusion data...
    load /home/amirro/mircs/experiments/experiment_0015/feats_test2.mat
    feat_name = 'test';
    load(sprintf('/home/amirro/mircs/experiments/experiment_0015/feats_%s2.mat',feat_name));
    w = [1 1 1 1 -5 5 1 1 .5 5 -1 -5];
    L_det = load('/home/amirro/mircs/experiments/experiment_0012/lipImagesDetTest.mat');
    [F_test,classes_train,ticklabels] = feats_to_featureset2(feats_train,L_det);
    h_test = w*F_test;
    load ~/mircs/experiments/experiment_0015/regionData_test.mat
    imageIndices = [regionData_test.imageIndex];
    %     imageScores = L_det.newDets.cluster_locs(:,12);
  
    %%
end
L_det = load('/home/amirro/mircs/experiments/experiment_0012/lipImagesDetTest.mat');
detPath = '/net/mraid11/export/data/amirro/detectors_hog_s40/';
% r = dir(fullfile(detPath,'*.mat'));
conf.get_full_image = true;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
ids_ = test_ids;
labels_ = test_labels;
imageSet = imageData.test;

dataset_res = struct;
conf.get_full_image = true;

is = 1:length(ids_);
% test_scores = -inf*ones(size(ids_));
%%
for ik = 1:length(ids_)
    k = is(ik)
    if (imageSet.faceScores(k) < -.6)
        disp('skipping due to low face score');
        continue
    end
    
    if (~labels_(k))
        continue;
    end
    currentID = ids_{k};
    disp(currentID)
    resPath = fullfile(detPath,strrep(currentID,'.jpg','.mat'));
    if (~exist(resPath,'file'))
        disp('doesn''t exist - skipping');
        continue;
    end
    
    fisher_results_file = fullfile('/net/mraid11/export/data/amirro/res_s40_fisher',strrep(currentID,'.jpg','.mat'));
    if (~exist(fisher_results_file,'file'))
        disp('fisher file doesn''t exist - skipping');
        continue;
    end
    
    fisher_features_file = fullfile('/net/mraid11/export/data/amirro/occluded_seg_features_s40',strrep(currentID,'.jpg','.mat'));
    if (~exist(fisher_features_file,'file'))
        disp('fisher file doesn''t exist - skipping');
        continue;
    end
   
    
    occludersFile = fullfile('~/storage/occluders_s40',strrep(currentID,'.jpg','.mat'));
    if (~exist(occludersFile,'file'))
        disp('occluders file doesn''t exist - skipping');
        continue;
    end
    
    imgName = currentID;
   
%     pause;
%     continue;
    
%     L = load(resPath);

%     [regions,~,G] = getRegions(conf,currentID,false);
%     load(resPath,'regionRes'); % res with polys,class,scores,theta,rects
%     for kk = 1:length(res)
%         res(kk).polys = double(res(kk).polys);
%     end
%         
    % % %     B = load(fullfile('~/storage/objectness_s40',strrep(currentID,'.jpg','.mat')));
    % % %     objectnessMap = computeHeatMap(I,B.boxes,'sum');
    % % %     objectnessMap = normalise(objectnessMap);
        
    % get the potential regions for this image.
    [xy_c,mouth_poly,face_poly] = getShiftedLandmarks(imageSet.faceLandmarks(k),-I_rect);
    faceBox = imageSet.faceBoxes(k,:);
    faceBox = faceBox + I_rect([1 2 1 2]);
    faceBox_i = inflatebbox(faceBox,[1 1],'both',false);
    params.inflationFactor = 1.5;
    params.regionExpansion = 1;
    params.ucmThresh = .1;
    params.fullRegions = true;
%     [curRegions,groups,ref_box,face_mask,mouth_mask,I_sub,params] = extractRegions(conf,imageSet,k,params);

    imageIndex = findImageIndex(newImageData,currentID);
    imgData = newImageData(imageIndex);
%     [occlusionPatterns,dataMatrix,region_sel,face_mask] = getOcclusionPattern(conf,imgData);
               
    % load the occluding regions.
    
    
    load(occludersFile);
    
    L_fisher = load(fisher_features_file);
    regions = L_fisher.regions;
    feats = L_fisher.feats;
    
    
    
%     t = find(region_sel);
%     region_sel(t) = region_sel(t) & [occlusionPatterns.seg_in_face] >0 & ...
%     [occlusionPatterns.face_in_seg] < .5 & [occlusionPatterns.seg_in_face] < 1;


    
    
    
%     [curRegions,groups,ref_box,face_mask,mouth_mask,I_sub,params,region_sel] = extractRegions2(conf,imgData,params);
% % %     curRegions = curRegions(region_sel);
% % %     [~,curRegions] = expandRegions(curRegions,[],groups);
% % %     N = numel(curRegions{1});
% % %     areas = cellfun(@nnz,curRegions);
% % %     curRegions((areas/N) > .5) = [];
% % %     if (isempty(curRegions))
% % %         continue;
% % %     end
% % %     curRegions = fillRegionGaps(curRegions);
% % %     curRegions = col(removeDuplicateRegions(curRegions));
% % %     
% % % %     displayRegions(I,curRegions,zeros(size(curRegions)),.01);
% % % %     continue;
% % %     % %   
% % %     [curRegionFeats] = extractRegionFeatures(conf,curRegions,imgData,false,-.6,true);
% % %     if (isempty(curRegionFeats))
% % %         continue;
% % %     end
% % %     
    %     train_mouth_scores = L_det.newDets.cluster_locs([feats_train.imageIndex],12);
    curDet = L_det.newDets;
    %     curDet.cluster_locs=curDet.cluster_locs(curDet.cluster_locs(:,11)==k,:);
%     for z = 1:length(curRegionFeats), curRegionFeats(z).imageIndex = k; end
%     [f,c_,~] = feats_to_featureset2(curRegionFeats,[]);
    
%     w = [1 1 1 1 -5 5 1 1 .5 5 -1 -5];
%     h = w*f;
%     
%     if (isempty(curRegionFeats))
%         disp('no occluding regions found');
%         continue;
%     end
%                             
%     occludingRegions = {curRegionFeats.region};
%     occlusionScores = h;
%     feats = partModels(1).extractor.extractFeatures(currentID,occludingRegions);
%     [res_pred, res] = partModels(1).models.test(feats);
    
    [res_pred, res] = partModels(1).models.test(feats);
    [res_pred, res1] = partModels(2).models.test(feats);
    res = max(res,res1);
%     load (fisher_results_file);  
%     res = regionConfs.score;
    res(isnan(res)) = -inf;
%     regions = regions(region_sel);
    test_scores(k) = max(res);
    
%     continue;
    [I,I_rect] = getImage(conf,imgName);
    
    clf; subplot(2,2,1); imshow(I); axis image; hold on;
    displayRegions(I,regions,res,0,3)
%     regionScores = [regionRes.ovpScore].*[regionRes.hogScore];
%     displayRegions(I,occludingRegions,res)

    pause(.1);
    title('done, hit any key to continue'); pause
    continue;
    displayRegions(I,regions,regionScores,.1,30);
    
    regionBoxes = cellfun2(@(x) pts2Box(fliplr(ind2sub2(dsize(I,1:2),find(x)))), regions);
    regionBoxes = cat(1,regionBoxes{:});
    % % % %
    % show some results....
    
    
    
    %     L = load(fullfile('~/storage/boxes_s40',strrep(currentID,'.jpg','.mat')));
    %     regionBoxes = double(L.boxes(:,[1 2 3 4]));
    % %     regions = mat2cell2(regions,[size(regions,1) 1]);
    %     Z = computeHeatMap(I,[regions ones(size(regions,1),1)],'sum');
    polys = col({res.polys});
    scores = col([res.scores]);
    classes = col([res.class]);
    t_score = scores >= .15;
    
    polyBoxes = cellfun2(@pts2Box,polys);
    polyBoxes = cat(1,polyBoxes{:});
    
    
    poly_face_ovp = boxesOverlap(polyBoxes,faceBox_i);
    % want only polys which partially overlap face area (but not too large
    % overlap).
    t_ovp = poly_face_ovp > 0 & poly_face_ovp < .5;
    t_polys = t_ovp & t_score;
    %polys_ = polys(t_polys);
    %     scores_ = scores(t_polys);
    %     classes_ = classes(t_polys);
    polyBoxes = polyBoxes(t_polys,:);
    poly_ovp = boxesOverlap(polyBoxes,regionBoxes);
    [max_ovp,i_max_ovp] = max(poly_ovp,[],2);
    t_polys(t_polys) = t_polys(t_polys) & (max_ovp >= .5);
    
    i_max_ovp = i_max_ovp(max_ovp >= .5);
    poly_ovp = poly_ovp(max_ovp>=.5,:);
    max_ovp = max_ovp(max_ovp >= .5,:);
    
    %r = false(size(t_polys)); r(t_polys) = max_ovp;
    %t_polys = t_polys & r;
    polys_ = polys(t_polys);
    scores_ = scores(t_polys);
    classes_ = classes(t_polys);
    %     poly_gpb_ovp = boxesOverlap(polyBoxes,regionBoxes);
    %     poly_gpb_ovp = max(poly_gpb_ovp,[],2);
    
    %     [max_ovp,i_max_ovp] = max(poly_ovp,[],2);
    
    %     if (imageScores(k) > .6)
    %         disp('mouth score too high'); imagesc(I); axis image; pause(.3);
    %         continue;
    %     end
    
    
    for iClass = 1:length(classifiers)
        %         if (isfield(all_res{iClass},'poly'))
        b = classes_==iClass;
        if (none(b))
            disp(['no objects found for class ' classifiers(iClass).name ' - skipping.']);
            continue;
        end
        scores = scores_(b);
        polys = polys_(b);
        poly_ovps = poly_ovp(b,:);
        [s,is] = sort(scores,'descend');
        is = is(1:min(100,length(s)));
        polys = polys(is);
        poly_ovps = poly_ovps(is,:);
        HH = computeHeatMap_poly(I,polys,s(is),'max');
        
        % find the "top" of each poly.
        % we know this to be the first edge, by construction in
        % box2pts (a bit of a hack but saves me some other hassle)
        polys2 = cellfun2(@(x) x(1:2,:),polys);
        
        centers = cellfun2(@mean ,polys);
        centers2 = cellfun2(@mean ,polys2);
        centers = cat(1,centers{:});
        centers2 = cat(1,centers2{:});
        dd = centers2-centers;
        faceBox = imageSet.faceBoxes(k,:);
        faceBox = faceBox + I_rect([1 2 1 2]);
        faceCenter = mean([faceBox(1:2);faceBox(3:4)]);
        d_toface = bsxfun(@minus,faceCenter,centers);
        
        % show only the detection which "point" to the head, and are
        % sufficiently near.
        
        
        feats = getGeometricFeatures(polys,mouth_poly,face_poly);
        
        
        t1 = feats.intersection_of_faces < .4;
        t2 = feats.intersection_of_polys < .3;
        t3 = feats.ovp < .5;
        dist_to_face = sum(feats.poly_top_to_face.^2,2).^.5;
        dist_to_mouth = sum(feats.poly_top_to_mouth.^2,2).^.5;
        t4 = dist_to_face/feats.face_scale < 1;
        t5 = feats.poly_s_to_face_s < 2;
        t6 = sum(feats.poly_center_to_top_n.*feats.poly_center_to_face_n,2) > .7;
        t7 = dist_to_mouth/feats.face_scale < .5;
        t8 = sum(bsxfun(@times,feats.poly_center_to_top_n,[0 -1]),2) > -.1;
        
        t_debug = sum(bsxfun(@times,feats.poly_center_to_top_n,[1 0]),2) > .7 &...
            feats.poly_scales < 71 & feats.poly_scales > 69;
        %
        t_all = double([t1(:) t2(:) t3(:) t4(:) t5(:) t6(:) t7(:) t8(:) t_debug(:)]);
        
        w_ = [1 1 1 1 1 1 1 1 0]';
        %         w_ = [1 1 1 0 0 0 0 0 0]';
        %             w_ = [0 0 0 0 0 0 0];
        r = t_all*w_;
        
        r = r==sum(w_);
        if (none(r))
            continue;
        end
        polys = polys(r);
        
        
        polys2 = polys2(r);
        centers = centers(r,:);
        dd = dd(r,:);
        scores = scores(r);
        heatmap = computeHeatMap_poly(I,polys,scores,'max');
        II = sc(cat(3,heatmap,I),'prob');
        %clf; subplot(1,3,1); imagesc(I); axis image; hold on;
        clf; subplot(2,2,1); imagesc(II); axis image; hold on;
        
        subplot(2,2,2); imagesc(I); axis image; hold on;
        
        hold on; plotPolygons(polys,'m');
        hold on; plotPolygons(polys2,'g');
        
        plotPolygons({mouth_poly},'g','LineWidth',4);
        plotBoxes(faceBox,'m','LineWidth',3);
        
        quiver(centers(:,1),centers(:,2),dd(:,1),dd(:,2));
        
        title([classifiers(iClass).name num2str(max(heatmap(:)))]);
        
        %         II_with_obj = sc(cat(3,heatmap.*objectnessMap,I),'prob');
        %         subplot(2,2,3);imagesc(II_with_obj); axis image;
        %         title(num2str(max(max(heatmap.*objectnessMap))));
        II = sc(cat(3,HH,I),'prob');
        subplot(2,2,4); imagesc(II); axis image;
        %             subplot(1,3,3);
        %             displayRegions(I,regions(iovp(:,1)),ovp(:,1),0,1);
        
        % display the regions w. the overlap.
        poly_ovps = poly_ovps(r,:);
        
        for iPoly = 1:size(poly_ovps,1)
            iPoly
            rr = regions(poly_ovps(iPoly,:) > 0);
            [ovp,ints,areas] = boxRegionOverlap(poly2mask2(polys{iPoly},size(I)),rr);
            [ovp,iovp] = max(ovp);
            subplot(2,2,3); displayRegions(I,rr(iovp),ovp,0);
        end
        disp('...');
        %         subplot(2,2,3); imagesc(Z); axis image;
        %     imageScores(k)
        disp('hit any key');pause;clc
        %         end
    end
    disp('done! hit enter to move on to next image');
    %     imageScores(k)
    pause; clc;
    
end

[prec,rec,aps] = calc_aps2(test_scores',labels_)
plot(rec,prec)
% end
