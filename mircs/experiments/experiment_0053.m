
%% 13/1/2014
% zoom in on mouth region alone, to do the best you can there.
% now we have for each image it's
% 1. location of face
% 2. facial landmarks
% 3. segmentation
% 4. saliency
% 5. location of action object (pixel-wise mask)
% 6. prediction of location of action object, learned separately.
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    load fra_db.mat;
    all_class_names = {fra_db.class};
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    % make sure class names corrsepond to labels....
    [lia,lib] = ismember(classes,class_labels);
    classNames = all_class_names(lib);
    isTrain = [fra_db.isTrain];
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    initialized = true;
    conf.get_full_image = true;
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
    load ~/storage/mircs_18_11_2014/s40_fra
    params.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    load('~/storage/mircs_18_11_2014/allPtsNames','allPtsNames');
    [~,~,reqKeyPointInds] = intersect(params.requiredKeypoints,allPtsNames);    
    s40_fra_orig = s40_fra_faces_d;
    fra_db = s40_fra;
    net = init_nn_network();
    nImages = length(s40_fra);
    top_face_scores = zeros(nImages,1);
    for t = 1:nImages
        top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
    end
    min_face_score = 0;
    img_sel_score = col(top_face_scores > min_face_score);
    fra_db = s40_fra;
    top_face_scores_sel = top_face_scores(img_sel_score);
    
    % inialize parameters for various modules:
    % facial landmark parameters
    landmarkParams = load('~/storage/misc/kp_pred_data.mat');
    landmarkParams.kdtree = vl_kdtreebuild(landmarkParams.XX);
    landmarkParams.conf = conf;
    landmarkParams.wSize = 96;
    landmarkParams.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[landmarkParams.wSize landmarkParams.wSize],'bilinear')))) , y);
    landmarkParams.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    
    % segmentation...
    addpath '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';install;    
    clear predData;
    
    %% object prediction data
    %     load ~/code/mircs/s40_fra.mat;
    objPredData = load('~/storage/misc/actionObjectPredData.mat');
    objPredData.kdtree = vl_kdtreebuild(objPredData.XX,'Distance','L2');
    %
    %%
    load  ~/storage/misc/aflw_with_pts.mat; %  ims pts poses scores inflateFactor resizeFactors
end


maxImageSize = 128;
opts.maxImageSize = maxImageSize;
spSize = 150;
opts.pixNumInSP = spSize;
opts.show  =true;                            
totalSal = zeros(128);
totalSal_bd = zeros(128);
t = 0;
s = zeros(128,128,3);
figure(1)
range = 1:5:1000%:length(ims);
for u = range
%     u
    opts.show = false;
    s = s+(imResample(im2double(ims{u}),[128 128],'bilinear'))/length(range);
    [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(ims{u}),opts);
    t = t+1;
    totalSal = totalSal+ imResample(sal,size(totalSal),'bilinear');
    totalSal_bd = totalSal_bd+imResample(sal_bd,size(totalSal),'bilinear');
    if (mod(t,10)==0)
        u
        clf;
        vl_tightsubplot(2,2,1);imagesc2(totalSal);
        vl_tightsubplot(2,2,2);imagesc2(-totalSal_bd);
        vl_tightsubplot(2,2,3);imagesc2(normalise(s));
    %     [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(normalise(s)),opts);
    %     vl_tightsubplot(2,2,4);imagesc2(normalise(sal));
        drawnow
    end
%     pause
end

% extract all mouth regions....
%%
% fra_db(1).mouth
mouth_images = cell(size(fra_db));
tt = ticStatus('cropping mouths',.5,.5);
% scores = -inf(size(fra_db));

%%
for t = 1:1:length(fra_db)
    if ~fra_db(t).isTrain,continue,end
    if ~fra_db(t).valid,continue,end
    if fra_db(t).classID~=9,continue,end
    imgData = fra_db(t);
    I = getImage(conf,imgData);        
    [w h] = BoxSize(imgData.faceBox);
    mouthBox = inflatebbox([imgData.mouth imgData.mouth],w/2,'both',true);
    mouthBox = round(mouthBox);
    mouth_images{t} = cropper(I,mouthBox);
%     tocStatus(tt,t/nnz([fra_db.isTrain]));
%     continue    
    clf;vl_tightsubplot(1,2,1);
    imagesc2(I); zoomToBox(imgData.faceBox);
    plotPolygons(imgData.mouth,'g+');
    plotBoxes(imgData.faceBox);    
    vl_tightsubplot(1,2,2); imagesc2(imResample(mouth_images{t},[64 64]))
    
%     x2(I); plotBoxes(imgData.raw_faceDetections.boxes)
    
    % you were in the middle of checking mouth boxes...but this image,   imageID: 'smoking_062.jpg'
    %          imgIndex: 2901, has a weird face detection >> upon running
    %          the face detection, indeed I found that                 
    
    
%     plotBoxes(mouthBox);
    drawnow;
    pause
%     imgData.mouth
end

%%
trains = col([fra_db.isTrain]);
valids = col([fra_db.valid]);
class_labels = col([fra_db.classID]);
sel_ = trains & img_sel_score & class_labels~=conf.class_enum.DRINKING;
save ~/storage/misc/mouth_images2.mat trains valids class_labels mouth_images
load  ~/storage/misc/aflw_with_pts.mat; %  ims pts poses scores inflateFactor resizeFactors
mImage(mouth_images(sel_));
M = mouth_images(sel_);
M = cellfun2(@(x) imResample(x,[32 32],'bilinear'),M);
x2(M);

x2(M(100))
I1 = mouth_images{841};
x2(I1)

I1 = imResample(I1,2,'bilinear');

% E = edge(im2doub


I1 = I;
[candidates, ucm] = im2mcg(I1,'accurate',true); % warning, using fast segmentation...

x2(candidates.superpixels)
x2(I1)
% I1 = imcrop(I1);
[segImage,c] = paintSeg(I1,candidates.superpixels);
x2(segImage)
x2(ucm(1:2:end,1:2:end))
x2(I)
% masks = imageStackToCell(candidates.masks);

%%
mouth_centers = squeeze(getKPCoordinates_2(pts,{'MouthCenter'}));
%%
goods = ~isnan(mouth_centers(:,1));
sub_mouths = {};
for t = 1:length(ims)
    if (~goods(t)),continue,end
    t
    I = ims{t};
    curPts = pts(t);
    bb = [1 1 fliplr(size2(I))];
    bb = inflatebbox(bb ,1/1.3,'both',false);
    curMouthCenter = mouth_centers(t,:);
    [w h] = BoxSize(bb);
    mouthBox = inflatebbox([curMouthCenter curMouthCenter],w/2,'both',true);
    mouthBox = round(mouthBox);
    mouth_images{t} = cropper(I,mouthBox);
%     clf; imagesc2(I); plotPolygons(curPts.pts,'g+');
%     plotPolygons(mouth_centers(t,:),'r*');
%     plotBoxes(mouthBox,'r-');
%     plotBoxes(bb);
%     drawnow
%     pause
end
%%
f = find(isnan(mouth_centers(:,1)))
XX = getImageStackHOG(ims,[48 48],true,false,8);

% showNN(XX,ims,15,1)

%%