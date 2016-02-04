echo off;
if (~exist('toStart','var'))
    initpath;
    config;
    imageData = initImageData;
    toStart = 1;
    conf.get_full_image = true;
    imageSet = imageData.train;
    face_comp = [imageSet.faceLandmarks.c];
    cur_t = imageSet.labels;
    fb = FbMake(2,4,1);
    fb = squeeze(fb(:,:,3));
    dTheta = 10;
    thetaRange = 0:dTheta:180-dTheta;
    iv = 1:length(cur_t);
    allScores = -inf*ones(size(cur_t));
    frs = {};
    pss = {};
    hand_scores = {};
    f = find(cur_t);
    strawInds_ = f([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54]); % for train only!!
    strawInds = strawInds_;
    Zs = {};
    %     Zs_pos = {};
    %     Zs_neg = {};
    fhog1 = @(x) fhog(im2single(x),4,9,.2,0);
    
    % get the ground truth for cups...
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    [groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
        
    objInteractions = getObjectInteractions(groundTruth);
        
    ids1 = [objInteractions.id1];
    ids2 = [objInteractions.id2];    
    straw_and_cup = find(ids1 == 3 & ids2 == 1);
        
    straw_cup_images = {};
    
    for k = 1:length(straw_and_cup)
%         k
        curInteraction = objInteractions(straw_and_cup(k));
        I = getImage(conf,curInteraction.sourceImage);
        %bw1 = curInteraction.roi1 & curInteraction.roi2;
        R1 =  roipoly(I,curInteraction.roi1(:,1),curInteraction.roi1(:,2));
        R2 =  roipoly(I,curInteraction.roi2(:,1),curInteraction.roi2(:,2));
        
        % get the top of the cup roi, I don't need the straw here... 
%         
%         dist1 = bwdist(R1);
%         dist2 = bwdist(R2);
%         roi = dist1 <= 3 & dist2 <= 3;
%         %mm = imagesc(dist1+dist2);
        [yy,xx] = find(R1);        
        if (isempty(yy))
            continue;
        end
        mean_pt = mean([xx yy]);
        bb = inflatebbox([mean_pt mean_pt],[60 60],'both',true);
        straw_cup_images{end+1} = cropper(I,bb);
%         size( straw_cup_images{end})
%         clf; imagesc(I); axis image; hold on;
%         plotBoxes2(bb(:,[2 1 4 3]));
%         pause;
%         displayRegions(I,{roi});
        %clf; imagesc();
%         pause;        
    end
    
%     mImage(straw_cup_images);   
%     open learnBinaryFactors.m
           
%     X = allFeatures(conf,I);    
    gt_cup_aligned = alignGT_manual(conf,groundTruth,'cup');
%     cupImages = getGtImages(conf,gt_cup_aligned);

     cupImages = getGtImages(conf,gt_cup_aligned,true);

%     save cupImages cupImages
%  M =    mImage(cupImages);
imagesc(edge(rgb2gray(M),'canny'));
% find connected components...
%%
for k = 1:length(cupImages)
    clf;
    curImage = cupImages{k};
    %mm = min(1,size(curImage,1)/64);
    mm = 64/size(curImage,1);
%     curImage = imResample(curImage,mm);
    E = edge(rgb2gray(curImage),'canny');
    
    subplot(1,3,1);
    imagesc(curImage);axis image;
    subplot(1,3,2); imagesc(E); axis image;
    
    % make a grid....
    
    [seglist,edgelist] = processEdges(E);
    hold on;
    [M,O] = gradientMag(im2single(curImage));
    T = 3;
    
        
    for ii = 1:T
        n = size(E,1);
        E1 = zeros(size(E));
        range_ = floor(n*(ii-1)/T+1): floor(n*(ii)/T);
        E1(range_,:) = E(range_,:);
        [yy,xx] = find(E1);
                ellipse_t = fit_ellipse(xx,yy);
        if (isempty(ellipse_t) || isempty(ellipse_t.a))
            continue;
        end
        
        
        plot_ellipse(ellipse_t);
    end
    
%     for iEdge = 1:length(edgelist)
%         iEdge
%         yy = edgelist{iEdge}(:,1);
%         xx = edgelist{iEdge}(:,2);
%         %[yy,xx] = find(E);
%         ellipse_t = fit_ellipse(xx,yy);
%         if (isempty(ellipse_t) || isempty(ellipse_t.a))
%             continue;
%         end
%         
%         
%         plot_ellipse(ellipse_t);
%     end
     
    

    subplot(1,3,3); imagesc(M); axis image;
    pause;
    
end

%%
% try to find places that fit an ellipse


    cupImages2 = {};
    for k = 1:length(cupImages)
        C = imresize(cupImages{k},[NaN 96]);
        C = C(1:33,:,:);
        cupImages2{k} = C;
    end
    
    conf.features.vlfeat.cellsize = 8;
    X = imageSetFeatures2(conf,cupImages2,true,[]);
    
    [IDX,C] = kmeans2(X',10);  
%       clusters = makeClusters(X,[]);
    [clusters,ims] = makeClusterImages(cupImages2,C',IDX',X,'dir1');    
%     clusters = clusters(1);
    clusters = makeClusters(X(:,2),[]);
    save clusters clusters
    conf.features.winsize = size(vl_hog(im2single(cupImages2{2}),8));
    conf.detection.params.init_params.sbin = 8;
    conf.detection.params.max_models_before_block_method = 10;
    clusters_trained = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'suffix','cups_1','override',true);
    %clusters_trained = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','cups_1','override',false);    
    figure,imshow(showHOG(conf,clusters_trained(1).w))    
    figure,imshow(showHOG(conf,clusters(1).w))               
    [test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
%     profile off;

   %%
   
   for k = 1:length(cupImages)
       clf;
       I = max(0,min(1,cupImages{k}));
       subplot(2,1,1); imagesc(I); axis image;
       E = edge(rgb2gray(im2double(I)),'canny');
       subplot(2,1,2); imagesc(E); axis image;
       pause;
   end
   
   
    for q = 1:100
    TT = test_ids(test_labels);
    I = getImage(conf,TT{q});
%     imshow(I);
    
     [R,F]=vl_mser(255-rgb2gray(im2uint8(I)));
     F = vl_ertr(F);
%      [R,F]=vl_mser(rgb2gray(im2uint8(I)));
     
     clf;imagesc(I); hold on; axis image;
     vl_plotframe(F);
     pause;
    end
    
    I = imcrop(I);
    conf.detection.params.detect_min_scale = .1;
    conf.detection.params.detect_max_scale = 2;
    qq = applyToSet(conf,clusters_trained,{I},[],'cup_top_check','override',true,'rotations',[-20 0 20]);
    imshow('/home/amirro/code/mircs/cup_top_check.jpg');
% %     %%
    pause;
    end
%     profile viewer
    
%     multiWrite(cupImages,'~/cups_aligned');
                        
    partModelsDPM_cup = learnModelsDPM(conf,train_ids,train_labels,gt_cup_aligned,{'cup'});
    gt_bottle_aligned = alignGT_manual(conf,groundTruth,'bottle');
    partModelsDPM_bottle = learnModelsDPM(conf,train_ids,train_labels,gt_bottle_aligned,{'bottle'});
                     
    cupImages = getGtImages(conf,groundTruth,'cup');
%     cupImages = straw_cup_images;
%     mImage(cupImages);    
    x = {};
    cupImages1 = flipAll(cupImages);
    cupImages = [cupImages,cupImages1];
    conf.detection.params.init_params.sbin = 8;
    
    
    myFun = @(x) col(vl_hog(im2single(imresize(x,[64 48])),8));
    X = cellfun(myFun ,cupImages,'UniformOutput',false);
    X = cat(2,X{:});
                        
    conf.features.winsize = [8 6];
%     [X,uus,vvs,scales,t ] = allFeatures( conf,I,ovp )    
%     X = cat(2,x{:});
    load('hog_gauss');
    X1 = whiten(X,hog_gauss_data);
    
    X1_ = {};
    for k = 1:size(X1,2)
        x = reshape(X1(:,k),8,8,[]);
        X1_{k} = col(x(1:3,:,:));        
    end
    X1_ = cat(2,X1_{:});
    conf.features.winsize = [3 8];
    %%
    for q = 1:50
        q
        currentID = imageData.test.imageIDs{iv(q)};
        I = getImage(conf,currentID);
        %     imshow(showHOG(conf,X1(:,3)))
        conf.features.winsize = [4 9];
        [bbs,responses] = nnScan(conf,I,X1_);
        
        tops = {};
        for k = 1:size(responses,1)
            k
            tops{k} = esvm_nms([bbs(:,1:4),responses(k,:)'], 0);
        end
        tops= cat(1,tops{:});
        
        [tt,itt] = sort(tops(:,end),'descend');
        tops = tops(itt(1:100),:);
        
        I1 = computeHeatMap(I,tops,'sum');
        clf;
        subplot(1,2,1); imagesc(I); axis image;
        subplot(1,2,2); imagesc(I1); axis image; title(num2str(max(I1(:))));
        pause;
    end
    %%


    groundTruth = alignGT(conf,groundTruth);
    %     gtChoice =
    
    mImage(z_cups);
    
    z_cup_hog = {};
    for k = 1:length(z_cups)
        z = imresize(z_cups{k},[40 40]);
        z_cup_hog{k} = fhog(im2single(z(1:20,:,:)),4);
    end
    
    imagesc(hogDraw(z_cup_hog{3}.^2,15,1)); axis image; colormap jet
    
    zz = cat(4,z_cup_hog{:});
    z_templ = mean(zz(:,:,:,[2 5 8]),4);
    imagesc(hogDraw(z_templ.^4,15,1))
    images = getCanonizedGt(conf,groundTruth);
    
% end

addpath('/home/amirro/code/3rdparty/sliding_segments');
conf.demo_mode = true;

if (~exist('straw_train_data.mat','file'))
    [frs_train,pss_train,Zs_train,strawBoxes_train,xy_train, theta_train] = getStrawFeatures(conf,imageData.train,strawInds_,true);
    save straw_train_data frs_train pss_train Zs_train strawBoxes_train xy_train theta_train
else
    load straw_train_data;
end

posInds = strawInds;
negInds = find(~cur_t);

for k = 1:length(Zs_train)
    x = Zs_train{k};
    x(isnan(x(:))) = 0;
    Zs_train{k}=x;
end

Zs_train_pos = (cat(4,Zs_train{strawInds}));
ppos = fevalArrays(Zs_train_pos,fhog1);
sz = dsize(ppos,1:3);
ppos = reshape(ppos,[],size(ppos,4));
Zs_train_neg = cat(4,Zs_train{negInds});
pneg = fevalArrays(Zs_train_neg,fhog1);
pneg = reshape(pneg,[],size(pneg,4));
% gaborScores_train = getGaborScores(frs_train,pss_train);

[ws,b,sv,coeff,svm_model] = train_classifier(ppos,pneg(:,1:1:end),.001,50);

figure,imagesc(hogDraw(reshape(ws,sz),15,1));axis image;

% imageSubset = find(imageData.test.labels);
if (~exist('straw_test_data.mat','file'))
    [frs_test,pss_test,Zs_test,strawBoxes,xy_test,theta_test] = getStrawFeatures(conf,imageData.test);
    %     [frs_test,pss_test,Zs_test,strawBoxes] = getStrawFeatures(conf,imageData.test);
    save straw_test_data frs_test pss_test Zs_test strawBoxes xy_test theta_test
else
    load straw_test_data;
end
Zs_test = cat(4,Zs_test{:});
gabor_r_test = getGaborScores(frs_test,pss_test);
ptest = fevalArrays(Zs_test,fhog1);
ptest = reshape(ptest,[],size(ptest,4));
%%
ptestScores = ws'*ptest-b;
face_comp = [imageData.test.faceLandmarks.c];
imageSubset = find(cellfun(@(x) ~isempty(x),pss_test));
bad_scores = true(size(pss_test));
bad_scores(imageSubset) = false;
newScores = zeros(1,length(pss_test));
newScores(imageSubset) = ptestScores;
newScores = newScores+1*ismember(face_comp,6:11);
newScores = newScores + (1*imageData.test.faceScores>-.8);

gaborScores = [1 0 0]*gabor_r_test;
newScores = 1*newScores+10*real(gaborScores);
newScores(bad_scores) = min(newScores(~bad_scores));
[prec rec aps] = calc_aps2(newScores',imageData.test.labels,[],inf);
%%
% for debugging

[v,iv] = sort(newScores,'descend');
getStrawFeatures(conf,imageData.test,iv(1:end),true,ws,b,fhog1,[]);

%%
[v,iv] = sort(newScores,'descend');

for k = 1:length(iv)
    k
    %     iv(k)
    %     v(k)
    %     if (~imageData.test.labels(iv(k))) continue; end
    currentID = imageData.test.imageIDs{iv(k)};
    I = getImage(conf,currentID);
    clf;imagesc(I); hold on; axis image;
    pause;
end


%% get the cup detectore scores....

%% load all the confidence scores...
all_scores ={};
for q = 1:length(iv) % cup detector scores for this image....
    q
    k = iv(q);
    currentID = imageData.test.imageIDs{k};
    c_scores = load(fullfile('~/storage/res_s40/',strrep(currentID,'.jpg','.mat')));
    all_scores{k} = c_scores.regionConfs;
end

save all_scores all_scores

%% also load all region props...
all_props ={};
for q = 1:length(iv) % cup detector scores for this image....
    q
    k = iv(q);
    currentID = imageData.test.imageIDs{k};
    L_regions = load(fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat')));
    all_props{k} = L_regions.props;
end

save all_props all_props
%% check the combination of the classic object detection
debug_ = true;
cupDetScores = -1000*ones(size(iv));
handDetScores = -1000*ones(size(iv));
bottleDetScores = -1000*ones(size(iv));

for q = 1:length(iv) % cup detector scores for this image....
    q
    k = iv(q);
    currentID = imageData.test.imageIDs{k};
    %     c_scores = load(fullfile('~/storage/res_s40/',strrep(currentID,'.jpg','.mat')));
    %     L_regions = load(fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat')));
    props = all_props{k};
    cup_scores = all_scores{k}(1).score;
    hand_scores = all_scores{k}(2).score;
    bottle_scores =  all_scores{k}(4).score;
    
    bb = cat(1,props.BoundingBox);
    bb(:,[3 4]) = bb(:,[3 4])+bb(:,[1 2]);
    %     plotBoxes2(bb(:,[2 1 4 3]));
    %  get the face box..
    curFaceBox = imageData.test.faceBoxes(k,:);
    %     curFaceBox = strawBoxes{k};
    if (isempty(curFaceBox))
        continue;
    end
    expectedCupBox = [curFaceBox(1)-10,...
        (curFaceBox(2)+curFaceBox(4))/2,...
        curFaceBox(3),...
        2*curFaceBox(4)-curFaceBox(2)];
    expectedCupBox = inflatebbox(expectedCupBox,[2 1],'both');
    
    ints_ = BoxIntersection(expectedCupBox,bb);
    uns_ = BoxUnion(expectedCupBox,bb);
    [a_ b_ areas] = BoxSize(bb);
    [a_ b_ ints_] = BoxSize(ints_);
    [a_ b_ uns_] = BoxSize(uns_);
    goodBoxes = (ints_./areas) > .9;
    if (~any(goodBoxes))
        continue;
    else
        cupDetScores(k) = max(cup_scores(goodBoxes));
        handDetScores(k) = max(hand_scores(goodBoxes));
        bottleDetScores(k) = max(bottle_scores(goodBoxes));
    end
    if (debug_)
        I = getImage(conf,currentID);
        clf; imagesc(I); axis image; hold on;
        plotBoxes2(curFaceBox(:,[2 1 4 3]),'g-.','LineWidth',2);
        
        plotBoxes2(expectedCupBox(:,[2 1 4 3]),'r--','LineWidth',2);
        plotBoxes2(bb(goodBoxes,[2 1 4 3]));
        pause;
        continue;
    end
    %     [regions,regionOvp,G] = getRegions(conf,currentID,false);
    %     cupDetScores(k) = max(all_scores{k});
end

cupDetScores(cupDetScores==-1000) = min(cupDetScores(cupDetScores~=-1000));
handDetScores(handDetScores==-1000) = min(handDetScores(handDetScores~=-1000));
bottleDetScores(bottleDetScores==-1000) = min(bottleDetScores(bottleDetScores~=-1000));
%%
combinedScores = cupDetScores+1*handDetScores+1*bottleDetScores;
newScores_ = 1*newScores'+.1*combinedScores;
[prec rec aps] = calc_aps2(newScores_,imageData.test.labels,[],inf);

%% check the combination with top-of-cup detector
% fb = FbMake(2,5,1);
% fb = fb(:,:,1:2);
% montage2(fb)
% line_ = zeros(size(fb(:,:,1)));
% line_(8,:) = 1;
% fb_r = FbApply2d(line_,fb,'full',1);

debug_ = true;


cupDetScores = -1000*ones(size(iv));
conf.get_full_image = true;

useHOG = true;

for q = 1:length(iv) % cup detector scores for this image....
    q
    k = iv(q);
    if (imageData.test.labels(k))
        continue;
    end
    currentID = imageData.test.imageIDs{k};
    curFaceBox = strawBoxes{k};
      if (isempty(curFaceBox))
        continue;
    end
    expectedCupBox = [curFaceBox(1),...
        (curFaceBox(2)+curFaceBox(4))/2,...
        curFaceBox(3),...
        2*curFaceBox(4)-curFaceBox(2)];
    expectedCupBox = inflatebbox(expectedCupBox,[2 1],'both');
    expectedCupBox = round(expectedCupBox);
    I = getImage(conf,currentID);
    expectedCupBox = clip_to_image(expectedCupBox,[1 1 dsize(I,[2 1])]);
            
    I_sub = I(expectedCupBox(2):expectedCupBox(4),expectedCupBox(1):expectedCupBox(3),:);
        
%     getStrawFeatures(conf,imageData.test,iv(1:end),true,ws,b,fhog1,[]);
    
    resizeFactor = 128/size(I_sub,2);
    I_sub = imresize(I_sub,resizeFactor,'bilinear');
            
    I_sub = max(0,min(1,I_sub));
    
    if (useHOG)
    
    conf.detection.params.detect_keep_threshold = -1000;
    conf.detection.params.detect_max_windows_per_exemplar = inf;
    conf.detection.params.detect_min_scale = .5;    
    q_ = getDetections(conf,{I_sub},clusters_trained,[],'',false,...
        false,[0]);
    qq = getTopDetections(conf,q_,clusters_trained,'uniqueImages',false,...
        'useLocation',0,...
        'nDets',inf);
        
    qqq = cat(1,qq.cluster_locs);
        
    q_boxes = [round(qqq(:,[1:4])),qqq(:,12)];
    q_boxes(:,1:4) = clip_to_image(q_boxes(:,1:4),[1 1 dsize(I_sub,[2 1])]);
        
    I1 = computeHeatMap(I_sub,q_boxes,'max');
    end      
    curxy = xy_test{k};
    curxy_pt = mean(curxy');
    curTheta = theta_test{k};
    sind_ = sind(curTheta);
    cosd_ = cosd(curTheta);
    
    E = edge(im2double(rgb2gray(I)),'canny');
    
%     subplot(2,2,4); imagesc(E); axis image;
    xy_c = curxy_pt % clear all edges which are too far from the center point.
    [edge_y,edge_x] = find(E);
    f = l2([edge_x edge_y],xy_c) > (70^2);
    E(sub2ind(size(E),edge_y(f), edge_x(f))) = 0;
    [edgelist edgeim] = edgelink(E, []);
    seglist = lineseg(edgelist,3);
    seglist = [seglist,lineseg(edgelist,5)];
    seglist = segs2seglist(seglist2segs(seglist)); % break into single component segments.
    segs = seglist2segs(seglist);
    [EE,allPts] = paintLines(zeros(size(E)),segs);
    inds = makeInds(allPts);
    
    allPts = cat(1,allPts{:});
    dists = l2(xy_c,allPts).^.5;

    [m,im] = min(dists);
    segs_c = unique(inds(dists<5));
 
    [y,iy] = max(segs(segs_c,[1 3]),[],2);
    x = segs(sub2ind(size(segs),segs_c,iy*2));    

    [xy] = ([x y]-repmat(expectedCupBox(1:2),size(y,1),1))*resizeFactor;
     xy = round(xy);
     xy = xy(inImageBounds(dsize(I1,1:2),xy),:);
     vals = I1(sub2ind2(size(I1),xy(:,[2 1 ])));
     
     %goods = min(xy>0,[],2);
     
     if (~isempty(vals))
        cupDetScores(k) = max(vals);
     end
     
     
     responses = FbApply2d(rgb2gray(I),fb,'same');
     
   
      
    if (debug_)
        clf; subplot(2,2,1);imagesc(I); axis image; hold on;
 
        
        plotBoxes2(curFaceBox(:,[2 1 4 3]),'g-.','LineWidth',2);
        plotBoxes2(expectedCupBox(:,[2 1 4 3]),'r--','LineWidth',2);
        subplot(2,2,2); imagesc(I_sub); axis image;
        hold on;        
        plot(xy(:,1),xy(:,2),'m*');
             
        subplot(2,2,4);
        imshow(edge(im2double(rgb2gray(I_sub)),'canny')); axis image;
        
%         imagesc(max(abs(responses).^2,[],3));axis image;
%         hold on;        
%         plot(xy(:,1),xy(:,2),'m*');axis equal;
        
%         montage2(abs(responses));
        
%          vals = I1(sub2ind(size(I_sub),yy,xx));     
if (useHOG)
        subplot(2,2,4); imagesc(I1); axis image;title(num2str(max(vals(:))));
end
%         hold on;plot(xx,yy,'g.');
          
%         subplot(2,2,3); plot(vals);
        
        pause; 
    end      
end
%     [regions,regionOvp,G] = getRegions(conf,currentID,false);
%     cupDetScores(k) = max(all_scores{k});
cupDetScores(cupDetScores==-1000) = min(cupDetScores(cupDetScores~=-1000));
%%

combinedScores = cupDetScores;
newScores_ = 1*newScores'+1000*combinedScores;
[prec rec aps] = calc_aps2(newScores_,imageData.test.labels,[],inf);


%%
close all;
debug_ = true;

ff = find(cur_t);
% 505, the Obama image
% for k = ff([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54])'

% for q = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ] % 526
% iv = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ];
for q = 1:length(cur_t)
    q
    k = iv(q);
    imageInd = k;
    if (~debug_)
        if (~isinf(allScores(k)))
            continue;
        end
    end
    
    currentID = imageSet.imageIDs{imageInd};
    %     if(~cur_t(k))
    %         continue;
    %     end
    
    %     if (~ismember(q,strawInds))
    %         continue;
    %     end
    %
    %     if (~theList(k))
    %         continue;
    %     end
    %
    %
    %     if(cur_t(k))
    %         continue;
    %     end
    
    
    curTitle = '';
    %      clf ;pause;
    I = getImage(conf,currentID);
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    lipRectShifted = imageSet.lipBoxes(imageInd,:);
    %     clear regions;
    %     [regions,regionOvp,G] = getRegions(conf,currentID,false);
    if (debug_)
        clf;
        subplot(2,3,1);
        imagesc(I); axis image; hold on;
        plotBoxes2(faceBoxShifted([2 1 4 3]));
        plotBoxes2(lipRectShifted([2 1 4 3]),'m');
    end
    box_c = round(boxCenters(lipRectShifted));
    
    % get the radius using the face box.
    [r c] = BoxSize(faceBoxShifted);
    boxRad = (r+c)/2;
    
    bbox = [box_c(1)-r/4,...
        box_c(2),...
        box_c(1)+r/4,...
        box_c(2)+boxRad/2];
    bbox = round(bbox);
    
    if (debug_)
        plotBoxes2(bbox([2 1 4 3]),'g');
    end
    
    if (any(~inImageBounds(size(I),box2Pts(bbox))))
        if (~debug_)
            allScores(k) = -10^6;
        end
        continue;
    end
    I_sub = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I_sub = imresize(I_sub,[50 50],'bilinear');
    I_sub = rgb2gray(I_sub);
    
    % apply the filter-bank and look for strong double responses...
    fr = abs(FbApply2d(I_sub,fb2,'same'));
    % clip borders.
    ddd = 3;
    fr(1:ddd,:,:) = 0;
    fr(end-ddd+1:end,:,:) = 0;
    fr(:,1:ddd,:) = 0;
    fr(:,end-ddd+1:end,:) = 0;
    if (debug_)
        %         clf;
        %         subplot(2,3,2); montage2(fr);
        
        subplot(2,3,2);
        imagesc(I_sub); axis image; hold on; colormap(gray);
    end
    
    % crop responses to ranges of angles.
    sz = size(I_sub);
    cx = sz(2)/2;
    cy = 1;
    [xx,yy] = meshgrid(1:sz(2),1:sz(1));
    theta = 180-180*atan2(yy-cy,xx-cx)/pi;
    maxRad = 15;
    radCap = ((yy-cy).^2+(xx-cx).^2).^.5 <= maxRad;
    fr2 = fr;
    wTheta = dTheta/2;
    %     z = zeros(length(thetaRange),1);
    doCap = true;
    II = imresize(rgb2gray(I(bbox(2):bbox(4),bbox(1):bbox(3),:)),...
        [100 100],'bicubic');
    ps = phasesym(II);
    ps = imresize(ps,size(I_sub));
    if (doCap)
        for qq = 1:length(thetaRange)
            if (thetaRange(qq) < 145 && thetaRange(qq) > 45)
                s = (theta <= thetaRange(qq)+wTheta) & (theta>=thetaRange(qq)-wTheta);
                s = imdilate(s,ones(1,9));
                %                             clf; imagesc(s); pause;
                fr2(:,:,qq) = fr(:,:,qq).*s.*radCap;
                %             z(qq) = max(radon(fr2(:,:,q),thetaRange(qq)));
            else
                fr2(:,:,qq) = 0;
            end
            %             fr2(:,:,q) = s;
        end
    end
    
    
    frs{k} = fr2;
    pss{k} = ps;
    %     frs{k} = bsxfun(@times,fr2,phaseSym);
    %     size(fr)
    c_ = bsxfun(@times,fr2,ps);
    
    c_sum = (squeeze(sum(sum(c_,1),2))).^2;
    c_sum = c_sum/sum(c_sum);
    
    [m,im] = max(c_(:));
    
    if (~debug_)
        allScores(k) = m;
    end
    [ii,jj,kk] = ind2sub(size(fr2),im);
    finalAngle = thetaRange*c_sum;
    %         cd_ = maxRad*cosd(180-finalAngle);
    %         sd_ = maxRad*sind(180-finalAngle);
    cd_ = maxRad*cosd(180-thetaRange(kk));
    sd_ = maxRad*sind(180-thetaRange(kk));
    R = rotationMatrix(pi*(180-thetaRange(kk))/180);
    mx = 1; my = 1;
    x = [-1 1 1 -1]*maxRad/mx; y =  [-1 -1 1 1]*maxRad/my;
    xy = R*[x;y];
    xy = bsxfun(@plus,xy,[jj;ii]);
    % map it back to the original image.
    bsize = segs2vecs(bbox); %(width,height)
    resizeFactor = bsize(2)/size(I_sub,1);
    xy = xy*resizeFactor;
    xy = bsxfun(@plus,xy,bbox(1:2)');
    
    %remap this area to a canonical frame.
    stepSize = 1/40;
    
    [X,Y] = meshgrid(0:stepSize:mx,0:stepSize:my);
    x_t = mx*[0 1 1 0]';
    y_t = my*[0 0 1 1]';
    T = cp2tform([x_t y_t],xy','affine');
    [X_,Y_]= tformfwd(T,X,Y);
    % %         figure(2),subplot(1,2,1);imagesc(I); axis image; colormap gray; hold on; plot(X_(:),Y_(:),'r.')
    
    Z = cat(3,interp2(I(:,:,1),X_ ,Y_,'bilinear'),...
        interp2(I(:,:,2),X_ ,Y_,'bilinear'),...
        interp2(I(:,:,3),X_ ,Y_,'bilinear'));
    Zs{k} = Z;
    if (debug_)
        
        subplot(2,3,2);
        title(num2str(m));
        hold on;
        
        quiver(jj,ii,...
            cd_,sd_,0,'g','LineWidth',3);
        hold on; plot(jj,ii,'r+');
        
        % crop out a region corresponding to this area: rotate the image
        % and crop around the center.
        
        
        subplot(2,3,3);imagesc(Z); axis image;
        p = pts2Box([X_(:) Y_(:)]);
        p = inflatebbox(p,2,'both');
        subplot(2,3,4); imagesc(I); axis(p([1 3 2 4]));
        hold on; plot(xy(1,[1:4 1]),xy(2,[1:4 1]),'LineWidth',2);
        Z = im2single(Z);
        %         [f,d] = vl_sift(Z,'Frames',[21 21 1 0]','Orientations');
        
        %Zs_pos{end+1} = Z;
        %         Zs_neg{end+1} = Z;
        subplot(2,3,3); hold on;
        %         hold on; vl_plotsiftdescriptor(d,f);
        %         subplot(2,3,5); imagesc(imrotate(Z,180*f(4)/pi-90,'bilinear','crop')); axis image;
        ff = fhog1(Z);
        subplot(2,3,6); imagesc(hogDraw(ff.^2,15,1));
        %         title(num2str(ws'*ff(:)-b));
        subplot(2,3,5); montage2(c_);
        pause;continue;
        xy = [xy,xy(:,1)];
        plot(xy(1,:),xy(2,:),'m+-');
        %rr = imrotate(I_sub,180-thetaRange(kk),'bicubic');
        %         saveas(gcf,['/home/amirro/notes/images/drink_mirc/straw_new/' sprintf('%03.0f.jpg',q)]);
        pause;
        %         pause(.001);
        %         if (q==74)
        %             break
        %         end
        %
    end
    
    
    %     handsFile = fullfile(conf.handsDir,strrep(currentID,'.jpg','.mat'));
    %     L_hands = load(handsFile);
    %     bboxes = L_hands.shape.boxes;
    %     b_scores = bboxes(:,6); % remove all but top 1000
    %     [b,ib] = sort(b_scores,'descend');
    %     bboxes = bboxes(1:ib(min(length(ib),1000)),:);
    %     bb = nms(bboxes,.3);
    % %     hold on;
    % %     plotBoxes2(bboxes(bb,[2 1 4 3]));
    %     map = computeHeatMap(I,bboxes(bb,[1:4 6]),'max');
    %
    %     hand_scores{k} = mean(map(bbox(2):bbox(4),bbox(1):bbox(3)));
    
    continue;
    L_feats = load(fullfile('~/storage/bow_s40_feats/',strrep(currentID,'.jpg','.mat')));
    feats = (vl_homkermap(L_feats.feats, 1, 'kchi2', 'gamma', 1));
    
    % get a different region subset for each region type.
    
    rs = zeros(length(partNames),size(feats,2));
    for iPart = 1:length(partNames)
        [res_pred, res] = partModels(iPart).models.test(feats);
        rs(iPart,:) = row(res);
    end
    rs(isnan(rs)) = -inf;
    subsets = suppresRegions(regionOvp,1,rs,I,regions);
    
    % unite all subsets
    sel_ = unique([subsets{:}]);
    rs = rs(:,sel_);
    regions = regions(sel_);
    G = G(sel_,sel_);
    selBox = lipRectShifted;
    faceBox = faceBoxShifted;
    regionConfs = struct('score',{});
    for ii = 1:length(partNames)
        
        regionConfs(ii).score = rs(ii,:);
    end
    f = faceBoxShifted;
    cupBox = [f(1) f(4) f(3) f(4)+f(4)-f(2)];
    cupBox = inflatebbox(cupBox,[1 1],'both');
    hold on; plotBoxes2(cupBox([2 1 4 3]),'g');
    
    
    % %         pause;continue
    % % %
    %
    %
    %     displayRegions(I,rsegions,regionConfs(3).score,0,1);
    %
    %     continue;
    % 1. straw candidates: long object intersecting the lip area.
    L_regions = load(fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat')));
    props = L_regions.props(sel_);
    
    % find all regions overlapping with lips area.
    %     [lip_ovp1,lip_int1] = boxRegionOverlap(lipRectShifted,regions,[]);
    
    regionBoxes = cat(1,props.BoundingBox);
    regionBoxes(:,3:4) = regionBoxes(:,3:4)+regionBoxes(:,1:2);
    selBox = cupBox;
    [sel_ovp,sel_int,sel_areas] = boxRegionOverlap(selBox,regions,[],regionBoxes);
    
    [face_ovp,face_int,face_areas] = boxRegionOverlap(faceBoxShifted,regions,[],regionBoxes);
    % find the best face region...
    
    [r,ir] = sort(face_ovp,'descend');
    
    clf;
    %     imagesc(blendRegion(I,regions{ir(1)},-1)); axis image;
    b1 = blendRegion(I,regions{ir(1)},-1);
    L = load(fullfile('~/storage/boxes_s40',strrep(currentID,'.jpg','.mat')));
    regions_ss = double(L.boxes(:,[1 2 3 4]));
    
    
    bbovp = boxesOverlap(faceBoxShifted,regions_ss);
    [r,ir] = sort(bbovp,'descend');
    
    imagesc(blendRegion(b1,computeHeatMap(I,[regions_ss(ir(1),:) 1]),-1,[0 0 1])); axis image;
    
    hold on; plotBoxes2(regions_ss(ir(1:min(5,length(ir))),[2 1 4 3]),'m-.');
    
    %displayRegions(I,regions_ss,bbovp,5);
    
    hold on;plotBoxes2(faceBoxShifted([2 1 4 3]),'g-.','LineWidth',2);
    
    % find regions overlapping with the face, and within them, find a
    % better face region.
    
    %     new_bbovp = boxRegionOverlap(regions_ss(ir(1),:),[ggg,regions],[],[new_bb;regionBoxes]);
    %     pause
    %     displayRegions(I,[ggg,regions],new_bbovp,0,5)
    %
    pause;
    continue;
    [~,~,s] = BoxSize(round(faceBoxShifted));
    
    sel_inside = sel_int./sel_areas;
    lambda = 0;
    cup_score = (regionConfs(2).score)-lambda*face_int/s; % don't want segments in face. % + lambda*(sel_inside > .5);
    % also, don't want the area to be larger than the face :
    cup_score = cup_score;%((face_areas/s) > 0);
    cup_score(sel_ovp==0) = -inf;
    pause;
    displayRegions(I,regions,cup_score,0,1);
    continue;
    %     selBox = inflatebbox(lipRectShifted,2,'both');
    % find regions intersecting with lip area
    %[lip_ovp,lip_int,lip_areas] = boxRegionOverlap(selBox,regions,[],regionBoxes);
    % now, remove regions which start too high, e.g, intersect the region
    % above.
    antiSelBox = [1,1,size(I,2),selBox(2)];
    [anti_ovp] = boxRegionOverlap(antiSelBox,regions,[],regionBoxes);
    
    %lip_score = exp(-[props.MinorAxisLength]/5) + [props.Eccentricity]+10*[props.MajorAxisLength]/mean(dsize(I,1:2));
    lip_score = ([props.MinorAxisLength] < 10) + [props.Eccentricity]+10*[props.MajorAxisLength]/mean(dsize(I,1:2));
    lip_score = lip_score + 10*regionResponses;
    lip_score(~(lip_ovp > 0 & anti_ovp == 0 & lip_areas < 1000)) = 0;
    
    if (debug_)
        f =  find(lip_score);
        [q,iq] = sort(lip_score(f),'descend');
        showSel_ = f(iq(1:min(3,length(iq))));
        regionSubset = fillRegionGaps(regions(showSel_));
        displayRegions(I,...
            regionSubset, q);
        
    end
    continue;
    [parts,allRegions,scores] = followSegments3(conf,regions,G,regionConfs,I,selBox,faceBox,regionOvp,[],[]);
    
    allScores(k) = allRegions{1}(2);
    %         [],[]);%relativeShape,relativeShapes_);
    %     Z = zeros(dsize(I));
    %     for pp = 1:length(parts{1})
    %         Z(regions{allRegions{1}(pp)reg = pp;
    %     end
    
    %     pause;
    %
end

%%


% new_r = getGaborScores(frs_train,pss_train

new_r = zeros(size(allScores));
rads = ((yy-cy).^2+(xx-cx).^2).^.5;
rad_i = [0 15 30 45];
rr = zeros(3,18,length(new_r));
for k = 1:length(new_r)
    k
    fr2 = frs{k};
    ps = pss{k};
    if (isempty(fr2))
        continue;
    end
    %for qq = 1:length(thetaRange)
    for iRad = 2:length(rad_i)
        curRange = rads < rad_i(iRad) & rads>=rad_i(iRad-1);
        m = bsxfun(@times,fr2,curRange.*(ps.^.5));
        %         clf; imagesc(curRange); pause;
        %m = bsxfun(@times,fr2>0,curRange.*(ps));
        %         m = bsxfun(@plus,fr2,ps);
        %         m = bsxfun(@plus,fr2,.1*ps);
        %         m = bsxfun(@times,m,curRange);
        
        %m = bsxfun(@plus,m,ps);
        rr(iRad-1,:,k) = max(max(m,[],1),[],2);
        %             z(qq) = max(radon(fr2(:,:,q),thetaRange(qq)));
        %             fr2(:,:,q) = s;
    end
end


%%

new_r = squeeze(max(rr,[],2));
new_r1 = (new_r./squeeze(sum(rr,2)+eps));
hh = cellfun(@mean,hand_scores);
hh(isnan(hh)) = -100;
hh(isinf(hh)) = -100;
% new_r1(isnan(new_r1)) = 0;
% new_r = squeeze(max(rr,[],1));
% new_score = sum(new_r);
% new_r2 = sum(squeeze(sum(rr,2)))';
new_score = 1*(new_r(1,:)+0*new_r(2,:))+0*new_r(3,:);%-(hh>.3);
% new_score = sum(new_r(1,:))+0*.5*new_r(2,:)+0*.1*new_r(3,:);
% new_score = new_r(1,:)+0*new_r(2,:);
new_score = new_score+ismember(face_comp,6:11);
new_score = new_score + 0*(imageSet.faceScores>-.8);

% new_t = zeros(size(cur_t));
% new_t([ [563 556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ]]) = true;
% new_t = theList';

% [prec rec aps] = calc_aps2(new_score'new_score,cur_t,sum(test_labels));
[prec rec aps] = calc_aps2(new_score',cur_t);
% new_score = hh;
[v,iv] = sort(new_score,'descend');
%%
plot(1:length(v),v,'b');
hold on; plot(1:length(v),v.*cur_t(iv)','r+');
markMode = false;
if (markMode)
    theList = false(size(iv));
end

for k = 1:length(iv)
    iv(k)
    %     v(k)
    if (markMode)
        if (~cur_t(iv(k)))
            continue;
        end
    end
    currentID = imageSet.imageIDs{iv(k)};
    I = getImage(conf,currentID);
    clf; subplot(1,2,1); imagesc(I); hold on; axis image;
    
    
    subplot(1,2,2); montage2(frs{iv(k)});axis image
    
    
    if (markMode)
        t = getkey;
        if (t==30)
            theList(iv(k)) = true;
        elseif (t== 28 || t==27) % left, escape
            break;
        end
    else
        pause;
    end
end


%%
% save ZZ Zs_pos Zs_neg
ppos = fevalArrays(cat(4,Zs{strawInds}),fhog1);
sz = dsize(ppos,1:3);
ppos = reshape(ppos,[],length(strawInds));
negInds = find(~cur_t);
% nElements = sum(cellfun(@(x) ~isempty(x),Zs(negInds)));
pneg = fevalArrays(cat(4,Zs{negInds}),fhog1);
pneg = reshape(pneg,[],size(pneg,4));


[ws,b,sv,coeff,svm_model] = train_classifier(ppos,pneg);

imagesc(hogDraw(reshape(ws,sz),15,1));

% imagesc(HOGpicture(reshape(-ws,[10,10,31]),15))

