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
    fhog1 = @(x) fhog(im2single(x),4,9,.2,0);
    conf.demo_mode = true;
end


getStrawFeatures2(conf,imageData.train,strawInds_,true);
getStrawFeatures2(conf,imageData.train,zz(1:55),true);

% 1% r = [minorAxis,
% 2% majorAxis,
% 3% aspects,
% 4% repmat(wh(1),size(majorAxis)),
% 5% minorAxisRatio,
% 6% orientations,
%7sel_,
% 8% k*ones(size(minorAxis))];
M = getRegionFeatureMatrix(imageData.train);
% save M_train.mat M;

M = cat(1,M{:});
minorAxis = M(:,1); majorAxis = M(:,2); aspects = M(:,3); 
w = M(:,4); minorAxisRatio = M(:,5);
orientations = M(:,6);
sel_ = M(:,7) > 0;

scores = (majorAxis > 5) .* (aspects > 3) .* (abs(orientations > 40)) .* (minorAxisRatio < .1);
scores = scores.*sel_;
inds = M(:,8);
[r,ir] = sort(scores,'descend');
curInd = 0;
q = 0;

%[a b c] = unique(inds(ir));

zz = {}; % record the first occurence in each image
visited_ = false(size(imageData.train.imageIDs));
for k =1:length(inds)
    curInd = inds(ir(k));
    if (~visited_(curInd))
        zz{end+1} = curInd;
        visited_(curInd) = true;
    end
end

zz = [zz{:}];

for k = 1:length(ir)
    
    nextInd = inds(ir(k));
    if (nextInd == curInd)
        continue;
    else
        curInd = nextInd;        
    end
%     k
% q = q+1
    [q r(k)]
    if (r(k) == 0)
        break;
    end
    q = q+1;
    clf; imagesc(getImage(conf,imageData.train.imageIDs{nextInd}));
    pause();
end


I_subs_train = getSubMouthImages(conf,imageData.train);
I_subs_test = getSubMouthImages(conf,imageData.test);

save ~/storage/mouthsub.mat I_subs_train I_subs_test

fb2 = makeDoubleFilterBank;
cp = getAngleCaps([50 50],0:10:170);
cp_ = repmat(cp,[1 1 4]);

debug_ = true;

for k = 1:length(imageData.train.imageIDs) 
    k
    I = I_subs_train(:,:,strawInds_(k));
    
    fr = FbApply2d(im2double(I),fb2,'same');   
    fr_ = abs(fr.*cp_);
        
%     [ii,jj,kk] = max(
    if (debug_)
        clf; subplot(1,3,1); imagesc(I);    axis image;
        subplot(1,3,2); montage2(fr_);
        subplot(1,3,3);
        montage2(fb2);
           pause    
    end     
end


debug_ = false;

ms = zeros(1,length(imageData.test.imageIDs));
for k = 1:length(imageData.test.imageIDs) 
    k
    I = im2double(I_subs_test(:,:,k));
    
    fr = FbApply2d(I,fb2,'same');   
    fr_ = abs(fr.*cp_);
    ms(k) = max(fr_(:));
%     [ii,jj,kk] = max(
    if (debug_)
        clf; subplot(1,3,1); imagesc(I);    axis image;
        subplot(1,3,2); montage2(fr_);
        subplot(1,3,3);
        montage2(fb2);
           pause    
    end     
end

getStrawFeatures(conf,imageData.train,strawInds_,true);


if (~exist('straw_train_data.mat','file'))
    [frs_train,pss_train,Zs_train,strawBoxes_train,xy_train, theta_train] = getStrawFeatures(conf,imageData.train);%,strawInds_,true);
            
    save straw_train_data frs_train pss_train Zs_train strawBoxes_train xy_train theta_train
else
    load straw_train_data;
end

posInds = strawInds;
negInds = find(~cur_t);

D = zeros([64 64],'single');
D(:,32-4:32+4) = 1;
% D = repmat(D,3,1);
% imshow(D);


H = {};
for ik = 1:10
    ik
    k = (ik-1)*10;
D1 = imrotate(D,k,'bilinear','crop');
x = vl_hog(D1,8);
x = x(2:end-1,2:end-1,:);
V = vl_hog('render',x);
clf; imagesc(V);
pause
H{ik} =x(:);

end
H = cat(2,H{:});

imshow(I);

conf.features.winsize = [6 6];
clusters = makeClusters(double(H),[]);
clusters_trained = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'suffix','cups_1','override',true);

for k = 1:length(clusters)
    imshow(showHOG(conf,clusters_trained(k)));
    pause;
end

conf.detection.params.max_models_before_block_method = 10;
conf.detection.params.detect_add_flip = false;
conf.detection.params.init_params.sbin = 4;
conf.detection.params.detect_keep_threshold = -1000;
conf.detection.params.detect_max_windows_per_exemplar = 1000;
%%
curID = imageData.train.imageIDs{strawInds_(2)};
I = getImage(conf,curID);

conf.detection.params.detect_exemplar_nms_os_threshold = 0.9;
conf.detection.params.detect_min_scale = .1;
q = getDetections(conf,{I},clusters_trained,[],[],false,...
    false,[0]);
qq = getTopDetections(conf,q,clusters,'uniqueImages',false,...
    'useLocation',0,'nDets',inf);

% figure,imshow(I); hold on; plotBoxes2(qq(1).cluster_locs(:,[2 1 4 3]));

%%
for kk = 1:length(clusters_trained)
     (ik-1)*10
qqq = qq(kk).cluster_locs;
bb = [ round(qqq(:,[1:4])) qqq(:,12)];
bb = clip_to_image(bb,[1 1 dsize(I,[2 1])]);

bb = bb(1:min(1000,size(bb,1)),:);

I1 = computeHeatMap(I,bb,'sum');
clf;subplot(1,2,1); imagesc(I);axis image;
subplot(1,2,2); imagesc(I1.^2); axis image;
pause;
end
%%

%V = hogDraw(x,15,1);
% figure,imagesc(V);

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

gaborScores = [1 1 0]*gabor_r_test;
newScores = 1*newScores+10*real(gaborScores);
newScores(bad_scores) = min(newScores(~bad_scores));
[prec rec aps] = calc_aps2(newScores',imageData.test.labels,[],inf);

save newScores newScores
%%

% newScores = ms;
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

%% check the objectness at the end of the straw...

debug_ = true;


cupDetScores = -1000*ones(size(iv));
conf.get_full_image = true;
% obj_ = zeros(size(iv));


for q = 1:length(iv) % cup detector scores for this image....
    q
    k = iv(q);
    if (~imageData.test.labels(k))
%         continue;
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
    
    % objectness....
    B = load(fullfile('~/storage/objectness_s40',strrep(currentID,'.jpg','.mat')));
    %    
    
    [map,counts] = computeHeatMap(I,B.boxes(1:1000,:),'sum');
%         continue

    propsFile = fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat'));
    load(propsFile);
    segmentBoxes = cat(1,props.BoundingBox);
    segmentBoxes = imrect2rect(segmentBoxes);
    ovp = boxesOverlap(expectedCupBox,B.boxes)';
    obj_(k) = ovp'*B.boxes(:,5);
%     boxScores = zeros(size(ovp));
%     E = round(expectedCupBox);
%     boxObjectness = sum(sum(map(E(2):E(4),E(1):E(3))));
%     obj_(k) = boxObjectness;
%     for b = 1:length(boxScores)
%         boxScores(b) = 
%     end
        
    if (debug_)
        clf
        subplot(2,2,1); imagesc(I);
        hold on; plotBoxes2(expectedCupBox(:,[2 1 4 3]),'g','LineWidth',2);
        subplot(2,2,2); imagesc(map);                       
        pause; 
    end
    
    
    
end
%%
[prec rec aps] = calc_aps2(newScores'+.08*obj_',imageData.test.labels,[],inf);
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

    %    imagesc(map); axis image
    %
%         Z = imfilter(map,fspecial('gauss',25,8));
%    subplot(1,2,2);   imagesc(Z); axis image; 
% pause; continue;
% %     
%     z = cellfun(@(x) mean(Z(x)),regions);
%     z = z/max(z(:))-1*ovp_neg';
%     displayRegions(I,regions(sel_),z(sel_),0);
