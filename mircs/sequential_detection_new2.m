echo off;
if (~exist('toStart','var'))
    initpath;
    config;
    imageData = initImageData;
    toStart = 1;
    conf.get_full_image = true;
    %faceModel = learnFaceModel_(conf);
    imageSet = imageData.test;
    cur_t = imageSet.labels;
    fb = FbMake(2,5,1);
%     initUnaryModels;
end

allScores = -inf(size(cur_t));
%
%
%%
close all;
debug_ = true;
displayMode = true;
% 505, the Obama image
% for k = 1:len
for k = 1:length(cur_t)
%      for k = [ 549  551 552 556 476 587 492 496 497 505 506 519 525 540 546 ] % 526
%    k = iq(q);
    imageInd = k;
        
    if (~displayMode)
        if (~isinf(allScores(k)))
            continue;
        end
    end
    currentID = imageSet.imageIDs{imageInd};
    if (isempty(strfind(currentID,'drinking')))
        continue;
    end
    clc;
    disp(currentID);
    
    if(~cur_t(k))
%         continue;
    end
    curTitle = '';
%     fbRun
    I = getImage(conf,currentID);
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    lipRectShifted = imageSet.lipBoxes(imageInd,:);
    ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
    L = load(ucmFile);
    [regions,regionOvp,G] = getRegions(conf,currentID,false);
    
    if (debug_)
        clf;
        subplot(2,2,1);imagesc(I); axis image;hold on;
        
        plotBoxes2(faceBoxShifted([2 1 4 3]));
        lipRectShifted = inflatebbox(lipRectShifted,2,'both');
        plotBoxes2(lipRectShifted([2 1 4 3]));
        subplot(2,2,2);
        imagesc(L.ucm); axis image;
        
        skinprob = computeSkinProbability(255*im2double(I));
        normaliseskinprob = normalise(skinprob) > 0.5;
        subplot(2,2,3);
        imagesc(normaliseskinprob); axis image;
        subplot(2,2,4);
        imagesc(skinprob); axis image;

        
        
        
%         displayRegions(I,regions(end:-1:1),[],.1);
        
%         I1 = cropper(I,lipRectShifted);
%         imagesc(I1); axis image
%         
%         FR = FbApply2d(rgb2gray(I1),fb,'full');
%         
%         imagesc(max(abs(FR),[],3));
% %         E = edge(rgb2gray(I),'canny');
% %         imshow(E); axis image; hold on;
%         top_k = 50; % keep only the top 5 segments for each detection...
        pause; continue;
    end
    
    
    
    
%     pause
%     continue;
    T_ovp = 1;
    region_sel = suppresRegions(regionOvp, T_ovp); % help! we're being oppressed! :-)
    region_sel = region_sel{1};
    
    % is there a long region here?
    
    regions = regions(region_sel);
    
    selBox = lipRectShifted;
    
    G = G(region_sel,region_sel);
    
    faceBox = faceBoxShifted;
    
    conf.demo_mode = false;
    
    [ovp_mouth] = boxRegionOverlap(lipRectShifted,regions);
    [ovp_face,ints,areas] = boxRegionOverlap(faceBoxShifted,regions);
                            
    % want the area inside the face region to be small compared to the
    % overall area.
    areas = cellfun(@nnz,regions);
    [~,~,s] = BoxSize(faceBoxShifted);
    ovp_score = (ovp_mouth>0) - 10*ovp_face - 10*ints./s-10*areas/numel(regions{1}); 
    %...10*ints./areas-10*areas/numel(regions{1});
    ovp_score(ovp_mouth==0) = -inf;
    
%     L_feats = load(fullfile('~/storage/bow_s40_feats/',strrep(currentID,'.jpg','.mat')));
%     feats = (vl_homkermap(L_feats.feats, 1, 'kchi2', 'gamma', 1));
%     cupPart = 1;
%     [res_pred, res] = partModels(cupPart).models.test(feats);
    
%     res = 2*res+1*ovp_score';
    res = ovp_score';
    res(isnan(res)) = -inf;
    [o,io] = sort(res,'descend');
    if (displayMode)
        allScores(k) = max(res);
    end
%    
%     displayRegions(I,regions(io(1:5)),res(io(1:5)),0);        
    
    %     ovp_score(ovp_face>.1) = -inf;
    if (debug_)
        [o,io] = sort(ovp_score,'descend');
        sel_ = 1:min(length(io),5);
        regions(io(sel_)) = fillRegionGaps(regions(io(sel_)));
        displayRegions(I,regions(io(sel_)),ovp_score(io(sel_)));
       
    end
%     
    
    % %     faceConfs = regionConfs(1).score;
    %     %       faceConfs = regionConfs(2).score; % cup
    %     [ff,iff] = sort(faceConfs,'descend');
    %
    %     [parts,allRegions,scores] = followSegments3(conf,regions,G,regionConfs,I,selBox,faceBox,regionOvp,[],[]);
    %
    %     allScores(k) = allRegions{1}(2);
    %         [],[]);%relativeShape,relativeShapes_);
    %     Z = zeros(dsize(I));
    %     for pp = 1:length(parts{1})
    %         Z(regions{allRegions{1}(pp)reg = pp;
    %     end
    %     pause;
    %
end
%%
% 
% [prec rec aps] = calc_aps2(allScores,cur_t)
% figure,plot(rec,prec)
% 
% [q,iq] =sort(allScores,'descend');
% for k = 1:length(iq)
%     clf; imagesc(getImage(conf,imageSet.imageIDs{iq(k)}));
%     pause
% end

