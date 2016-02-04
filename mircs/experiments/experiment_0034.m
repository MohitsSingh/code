%%%%%%% Experiment 34 - straws revisited %
%%%%%%% April 2, 2014

echo off;
if (~exist('initialized','var'))
    initpath;
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
    config;
    initialized = true;
    %     init_hog_detectors;
    conf.get_full_image = true;
    load ~/storage/misc/imageData_new;
    %         dpmPath = '/home/amirro/code/3rdparty/voc-release5';
    %     addpath(genpath(dpmPath));
    addpath(genpath('~/code/3rdparty/geom2d'));
    addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));
    addpath('/home/amirro/code/3rdparty/PolygonClipper');
    addpath('/home/amirro/code/3rdparty/guy/unary');
    addpath('/home/amirro/code/3rdparty/Gb_Code_Oct2012');
    addpath('/home/amirro/code/3rdparty/rp-master');
    addpath(genpath('/home/amirro/code/3rdparty/seg_transfer'));
    addpath('~/code/3rdparty/logsample');
    addpath('~/code/3rdparty/export_fig');
    addpath('/home/amirro/code/3rdparty/linecurvature_version1b/');
    landmarks = [newImageData.faceLandmarks];
    face_comp = [landmarks.c];
    newImageData = augmentImageData(conf,newImageData);
    m = readDrinkingAnnotationFile('train_data_to_read.csv');
    newImageData = augmentGT(newImageData,m);
    subImages = {newImageData.sub_image};
    % annotate straws....
    %     bbLabeler({'straw'},'/home/amirro/storage/data/drinking_extended/straw','/home/amirro/storage/data/drinking_extended/straw/straw_anno');
    addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));    
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    fisherFeatureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher    
end

conf.demo_mode = true;

%%
%mm = 3; nn = 3;
mm = 2; nn = 3;
if (0)
    close all;
    debug_ = false;
    mm = 2; nn = 3;
    %%
    load faceActionImageNames;
    %reallyLoad = valids;            
    
    %%
    boosts = zeros(size(newImageData));
    ticId = ticStatus([],.5,.1);
    sub_ims = {};
    sub_im_feats = {};
    curLabel = zeros(size(newImageData));
    valids = false(size(newImageData));
    for q = 1:length(newImageData)
        if (~img_sel(q)),newImageData(q).straw_score = -inf;continue;
        end
        newImageData(q).straw_score = -inf;
        tocStatus( ticId, q/length(newImageData));
        curImageData = newImageData(q);
        if (curImageData.faceScore<-.6),continue;end
        curLabel(q) = -1;
        %         if (curImageData.faceScore >=-.6)
        if (curImageData.label && curImageData.isTrain)
            curLabel(q) = 0;
            if (any(strfind(curImageData.extra.objType,'straw')))
                curLabel(q) = 1;
            end
        end
        %         end
        L = load(j2m(conf.segDataDir,newImageData(q)));
% %         L = load(j2m('/net/mraid11/export/data/amirro/s40_seg_data_2',newImageData(q)));
        segData = L.segData;
        sub_ims{q} = L.curSubIm;
        sub_im_feats{q} = L.fisherFeats;
        [d1,d2] = meshgrid(segData.lineScores);
        totalScore = segData.totalScores;
        newImageData(q).straw_score = max(totalScore(:));
        valids(q) = L.isValid;
        %         boosts(q) = segData.scoreBoost;
    end
    %
    %%    
    scores = [newImageData.straw_score];%*occScores;
    scores([newImageData.faceScore]<-.6) = -inf;    
    scores = scores;%-boosts;
    scores_test = scores;scores_train = scores;
    isTrain = [newImageData.isTrain];
    scores_test(isTrain) = -inf; scores_train(~isTrain) = -inf;    
    sortScores = scores_train;
    %     scores(scores < 0) = 0;
    %     scores(4000:end) = -inf;    
    [r,ir] = sort(sortScores,'descend');
    showSorted(subImages,sortScores,100);
    %     straw_scores = scores;
    %     save straw_scores  straw_scores
    %%
end

%% try occlusion with the new face probabilities...
load ~/storage/data/faceActionImage_masks obj_masks seg_masks
curInds = find(isValid & img_sel);
imgData = newImageData(curInds);
%%
% 
% Ms = {};
% for k = 1:length(curInds)
%     k
%     curImageData = imgData(k);        
%     %Ms{k} = getSubImage(conf,curImageData,1.5,true);
%     
%     
% end
% 
% multiWrite(Ms,'~/face_ims',{imgData.imageID});
% 

rprops = {};
for k = 1:length(regions)
    rprops{k} = regionprops(regions{k},'Area','Eccentricity','MajorAxisLength','MinorAxisLength',...
        'Orientation');
    rprops{k} = rprops{k}(1);
end
rprops = cat(1,rprops{:});
region_polys = cellfun2(@(x) fliplr(bwtraceboundary2(x)),regions);
%%
for k = 1:length(curInds)
    curImageData = imgData(k);    
    [regions,regionOvp,G] = getRegions(conf,curImageData.imageID,false);
    [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1.5,false);    
    curProb = obj_masks{k}{1};
    I_prob = sc(cat(3,double(curProb),M),'prob');
    clf;subplot(1,3,1);imagesc2(I_prob);
    face_box = round(face_box);    
    regions = cellfun2(@(x) cropper(x,face_box),regions);
    regions = removeDuplicateRegions(regions);
    max_vals = cellfun2(@(x) max(curProb(x(:))),regions);max_vals = [max_vals{:}];
    min_vals = cellfun2(@(x) min(curProb(x(:))),regions);min_vals = [min_vals{:}];
    toKeep = max_vals-min_vals >=.5;
    regions = regions(toKeep);min_vals = min_vals(toKeep);max_vals = max_vals(toKeep);
    subplot(1,3,2);                   
%     shapeFeats = getShapeFeatures(conf,M,regions);    
    displayRegions(M,regions,max_vals-min_vals,0,5);       
%     figure,plot(sort(curProb(regions{10}(:))))    
end


%% HERE
iv = ir;
% iv = 1:4000 
conf.straw.extent = 1;
conf.straw.dim = 100;
debug_ = true;
debug_2 = true;
only_extract_subwindows = true;
debug_seg_data = false;
checkOnlyPosClass = false;
conf.get_full_image = true
% iv = 1:4000
ddddd = debug_  && ~debug_2 && ~only_extract_subwindows && ~debug_seg_data;
for q = 1:length(iv) % 41 % 17 - very interesting, glasses pop out
    q
    k = iv(q)
%     k = 111
    %     k = 5230
    %findImageIndex(newImageData,'rowing_a_boat_100.jpg')
    %     k = 2680
%     if (~isValid(k)) % TODO - this is the isValid from exp. 36!!!
%         continue
%     end   
    curImageData = newImageData(k);    
%     if (~any(strfind(curImageData.imageID,'blow'))),continue,end    
    disp(curImageData.imageID)
    if (curImageData.faceScore < -.6), continue,end;
    if (checkOnlyPosClass)
        if (~curImageData.label), continue; end
        if (~any(strfind(curImageData.extra.objType,'straw'))),continue; end        
    end
    [I,I_rect] = getImage(conf,curImageData);
    f = j2m(conf.occludersDir, curImageData);
    L = load(f);            
    if (isempty(L.rprops)),seg_scores = -inf; else
        seg_scores = get_putative_occluders(conf,curImageData,L,I);
    end        
    % get the segments for the current face...
%     f_1 = j2m(conf.gpbDir_face,curImageData);
    f_1 = j2m('~/storage/gpb_s40_face_2',curImageData);
    L1 = load(f_1);
    [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1.5,true);
    face_mask = poly2mask2(face_poly,size2(I));
    faceScale = face_box(4)-face_box(2);                    
    face_mask= refineOutline_3(I,face_mask,ceil(faceScale/40)); %TODO -
%     face_mask = imerode(face_mask,ones(5));
    regions2 = L1.res.regions;
    perims = cellfun2(@(x) nnz(bwperim(x)),regions2);
    U = addBorder(false(size2(regions2{1})),1,1);
    % kill regions with too much border
    border_lengths = cellfun2(@(x) nnz(x & U),regions2);
    perims = cat(2,perims{:}); border_lengths = cat(2,border_lengths{:});
    regions2(border_lengths./perims > .2) = [];        
    regions2 = cellfun2(@(x) imResample(single(x),size2(M),'nearest'),regions2);
    regions2 = regions2(cellfun(@(x) nnz(x)>0,regions2));
    regions2 = removeDuplicateRegions(regions2);
%     displayRegions(M,regions2);
    regions2 = shiftRegions(regions2,round(face_box),I);
    % add more regions....    
    [regions,regionOvp,G] = getRegions(conf,curImageData.imageID);
    regions2 = [regions2,regions];         

    roi = round(inflatebbox(face_box,[1.5 1.5],'both',false));
    roi = clip_to_image(roi,I);
    curImageData.face_mask = face_mask;
    occlusionPattern = getOcclusionData(conf,curImageData,roi,regions2);  
    curImageData.occlusionPattern = occlusionPattern;
    curImageData.mouth_poly = mouth_poly;
    curImageData.face_poly = face_poly;
    
    [occludingRegions,occlusionPatterns,region_scores] = getOccludingCandidates_3(conf,I,curImageData);        

    clf;displayRegions(I,occludingRegions,region_scores,0,3);
    continue;
    occ_scores = score_occluders_new(curImageData,occludingRegions,occlusionPatterns);
    
    disp('displaying regions...');
   
        continue;
    %     maxOccScore = max(seg_scores);
    %[ucm,gpb_thin] = loadUCM(conf,curImageData.imageID);
%     addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');

%     [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1,true);
%     M = imResample(M,200/size(M,1),'bilinear');
%     segs = gpb_segmentation(conf,M);
%     displayRegions(M,segs.regions)
%     imagesc2(segs.ucm);figure,imagesc2(M)
% %     
    
%     continue;
            
    %     figure,imagesc2(I); hold on; %plotPolygons(xy_c,'r+');
    %     plotPolygons(mouth_poly,'r+');
    %     showCoords(xy_c);
    
    if (debug_ && debug_seg_data)
        segData = getSegData_debug(conf,curImageData,debug_);
    else
        L_1 = load(j2m(conf.segDataDir,curImageData));
        segData = L_1.segData;
    end
    if (debug_2)
        [curSubIm,windowPoly,orig_poly,target_rect_size] = extractObjectWindow(conf,curImageData,segData,debug_2);
        [regions,~,G] = getRegions(conf,curImageData.imageID,false);        
        target_mask = poly2mask2(orig_poly,size2(I));
        [res,best_ovp] = findCoveringSegment(regions,target_mask); 
        subplot(2,2,4);displayRegions(I,res);% pause;continue
        %[gc_segResult,obj_box] = checkSegmentation(conf,I,windowPoly,orig_poly);
        subplot(2,2,3);
        
        % add some more candidate regions.
        
%         [gc_segResult1,obj_box] = checkSegmentation(I,windowPoly,orig_poly);pause;
        % generally trust more the window-poly , without forcing the
        % orig_poly
        [gc_segResult2,obj_box] = checkSegmentation(I,windowPoly);pause;continue;
        
        regs = shiftRegions(gc_segResult,round(obj_box),I);
        reg_sub = rectifyWindow(regs,windowPoly,target_rect_size);
        % get only the largest component and keep it.
        %         gc_segResult = getSegments_graphCut_2(I,poly2mask2(windowPoly,size2(I)),[],1);
        %         figure,imagesc2(I)
        %          fisherFeatureExtractor.extractFeatures(curSubIm,reg_sub);
    end
    %sub_ims{k} =
    if (debug_2),pause;end
    if (only_extract_subwindows)
        continue
    end
    
    %     end
    %     J = textureMap(I,
    %
    if (debug_)
        %         L_1 = load(j2m(conf.segDataDir,curImageData));
        %         segData_1 = L_1.segData;
        
        %         segData = segData_1;
        totalScores = segData.totalScores;
        [a1,a2] = meshgrid(1:size(totalScores,1));
        totalScores(a1 <= a2) =-inf;
        [ii,jj,vv] = find(totalScores);
        ii = ii(vv>0);jj=jj(vv>0);
        vv = vv(vv>0);
        ij = [ii jj];
        if (isempty(vv))% || s(1) < 0)
            clf; imagesc2(I);
            disp('no good segments found');
            drawnow;pause;continue
        end
        [s,is] = sort(vv,'descend');
        [M,landmarks,face_box,face_poly,mouth_box,mouth_poly] = getSubImage(conf,curImageData,conf.straw.extent,true);
        s1 = size(M,1);
        sz = conf.straw.dim;
        M = imResample(M,[sz sz],'bilinear');
        
        % diversify the list...
        segs_as_list = segs2seglist(segData.segs(:,[2 1 4 3]));
        mouth_poly = fix_mouth_poly(mouth_poly);
        mouth_poly_2 = bsxfun(@minus,mouth_poly,face_box(1:2));
        mouth_poly_2 = mouth_poly_2*conf.straw.dim/s1;
        lineScores = segData.lineScores;
        pairScores = segData.pairScores;
        %ij = ind2sub2(size(totalScores),1:length(totalScores(:)));
        % create quadrangles from candidates.
        polygons = {};
        for kk = 1:length(is)
            s(kk)
            qq = is(kk);
            ii = ij(qq,1); jj = ij(qq,2);
            xy = reshape([segData.segs(ii,:)';segData.segs(jj,:)'],2,[])';
            xy = xy(convhull(xy),:);
            polygons{qq} = xy;
        end
        % find the intersections of all polygons
        A=cellfun2(@(x) poly2mask2(x,size2(M)),polygons);
        [ovp,ints,uns] = regionsOverlap(A,A);
        T_ovp = .5;
        subsets = suppresRegions(ovp,T_ovp,row(vv));
        subsets = cat(2,subsets{:});
        ij = ij(subsets,:);
        vv = vv(subsets);[s,is] = sort(vv,'descend');
        %         for iPoly = 1:length(polygons)
        %             for jPoly = iPoly+1:length(polygons)
        %                 intersectPolylines(
        %             end
        %         end
        
        for kk = 1:min(3,length(is))
            %             if (s(kk) < 0),break;end
            qq = is(kk);
            ii = ij(qq,1); jj = ij(qq,2);
            xy = reshape([segData.segs(ii,:)';segData.segs(jj,:)'],2,[])';
            
            
            xy = xy(convhull(xy),:);
            % shift the polygon back to the original image.
            xy = xy/(conf.straw.dim/s1);
            xy = bsxfun(@plus,xy,face_box(1:2));
            candidateMask = poly2mask2(xy,size2(I));
            
            if (~ddddd)
                subplot(mm,nn,4),imagesc2(candidateMask);
            end
            
            %check the intersection between this polygon and the major
            %occluders.
            occludingRegions = L.occludingRegions;
            candidateArea = nnz(candidateMask);
            portionsInFace = segData.props([ii jj],12);
            if (min(portionsInFace) == 1) % totally in face, check other occluders
                disp('totally inside face, checking occluders...')
                [ovp,ints,uns] = regionsOverlap(candidateMask,occludingRegions);
                score_boost = max(double(ints./candidateArea>0).*seg_scores);
                disp(['score boost: ' num2str(score_boost)]);
            end
            if (ddddd)
                clf; imagesc2(M);
            else
                subplot(mm,nn,6),cla; imagesc2(M);
            end
            drawedgelist(segs_as_list(ii),size2(M),2,[1 0 0]);
            drawedgelist(segs_as_list(jj),size2(M),2,[0 1 0]);
            plotPolygons(mouth_poly_2,'m--','LineWidth',2);
            title({sprintf('%d, (%d) score=%3.3f',kk,qq,s(kk)),...
                sprintf('score1(%d): %3.3f, score2(%d): %3.3f, pairscore: %3.3f',...
                ii,lineScores(ii),jj,lineScores(jj),pairScores(ii,jj)),...
                ['grad evergy: ' num2str(segData.gpb_reverse_evergy(ii,jj))],...
                ['line grad: ' num2str(segData.props([ii jj],7)')],...
                ['color diff: ' num2str(segData.color_diff(ii,jj))]});
            
            drawnow;
            %             export_fig(sprintf('/home/amirro/notes/images/2014_04_24/%s.pdf',curImageData.imageID));
            pause;
        end
    end
end

%%
doTrain = 0;
if (doTrain)
    
    train_set = ([newImageData.faceScore]>-.6) & [newImageData.isTrain] & valids;
    % [learnParams,conf] = getDefaultLearningParams(conf,1024);
    % fisherFeatureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
    % allFisherFeatures = extractFeatures_(fisherFeatureExtractor,sub_ims(train_set));
    poss = curLabel == 1 & train_set;
    negs = curLabel == -1 & train_set;
    
    features_pos = cat(2,sub_im_feats{poss});
%     load extraFeats
%     features_pos = [features_pos,morePosFeats];
    features_neg = cat(2,sub_im_feats{negs});
    classifier = train_classifier_pegasos(features_pos,features_neg,1);
    %%
    test_set =([newImageData.faceScore]>-.6) & ~[newImageData.isTrain] & valids;
    %allFisherFeatures_test = extractFeatures_(fisherFeatureExtractor,sub_ims(test_set));
    allFisherFeatures_test = cat(2,sub_im_feats{test_set});
    [~,h] = classifier.test(double(allFisherFeatures_test));
    % [~,h_flip] = classifier.test(double(features_test_flip));
    % h = classifier.w(1:end-1)'*features_test;
    showSorted(sub_ims(test_set),h,100);
    % showSorted(subImages(test_set),h,100);
        
    fisherFeatureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
    fisherFeatureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    fisherFeatureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
    test_ims = cellfun2(@(x) imResample(x,[120 120],'bilinear'),{newImageData(test_set).sub_image});
    test_feats = extractFeatures_(fisherFeatureExtractor,test_ims);
    load w;
    h = w(1:end-1)'*test_feats; 
%     allFisherFeatures = extractFeatures_(fisherFeatureExtractor,sub_ims(test_set));            
    %%
    h_all = -inf(size(test_set));h_all(test_set) = h;
%         h_all(scores_test<0) = -inf;
    h_final = h_all;
%     % showSorted(sub_ims,h_all,100);
%     h_final = 1000*h_all+scores_test;
    showSorted(subImages(test_set),h_final(test_set),50);
    [~,ir] = sort(h_all,'descend');
    %%    
    % %%
    % figure,imshow(I)
    % landmarks = detect_landmarks_99(conf,{I3},1);
    
end
if (0)
    %% extract features from the extended training set.
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    fisherFeatureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
    morePosFeats = {};
    moreSubWindows = {};
    imgDir ='/home/amirro/storage/data/drinking_extended/straw/';
    gtDir = fullfile(imgDir,'straw_anno');
    [filepaths,filenames] = getAllFiles(gtDir,'.txt');
    for k = 1:length(filepaths)
        k
        [objs,bbs] = bbGt('bbLoad', filepaths{k});
        if (isempty(objs)), continue;end
        bb = objs(1).bb;
        r1 = bb(3); r2 = bb(4);
        bb(3:4) = bb(1:2)+bb(3:4);
        
        I = imread(fullfile(imgDir,filenames{k}(1:end-4)));
        bb = rotate_bbs(bb,I,objs(1).ang,false);
        clf;subplot(1,2,1); imagesc2(I); plotPolygons(bb,'g','LineWidth',2);
%         pause;continue;
        %     r1 = .4; r2 = 1;
        
        %%%%%%%%

        asp_ratio = [.7 .3];
        r1 = asp_ratio(2); r2 = asp_ratio(1);
        ss = 200;
        target_rect_size = ss*[r1 r2];
        %%%%%%%%
        
%         ss = 80/r1; target_rect = round(ss*[r1 r2]);
        %ss = 200;target_rect = ss*[r1 r2]; .4*200
        obj_sub = rectifyWindow(I,bb{1},target_rect_size);
        moreSubWindows{end+1} = obj_sub;
        %morePosFeats{end+1} = extractFeatures_(fisherFeatureExtractor,obj_sub); %#ok<NASGU>
        morePosFeats{end+1} = fisherFeatureExtractor.extractFeatures(obj_sub); %#ok<NASGU>
%                     subplot(1,2,2); imagesc2(obj_sub);
%                     drawnow; pause;
    end
    morePosFeats = cat(2,morePosFeats{:});
    save extraFeats.mat moreSubWindows morePosFeats
    %%
    bbLabeler({'straw'},'/home/amirro/storage/data/drinking_extended/straw',gtDir);
    [gt0,dt0] = bbGt( 'loadAll', '/home/amirro/storage/data/drinking_extended/straw/straw_anno');
end