%% Experiment 0049 %%%%%
%% 8/9/2014
% Make my own facial landmark detector.
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    wSize = 96;
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
    
    load ~/storage/mircs_18_11_2014/s40_fra;
    fra_db = s40_fra;
    wSize = 32;
    extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[wSize wSize],'bilinear')))) , y);
    
    requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    addpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7'));
    initialized = true;
end

if (~exist('initialized2','var'))
    myDataPath = '~/storage/misc/kp_pred_data.mat';
    if (exist(myDataPath,'file'))
        load(myDataPath);
    else
        [paths,names] = getAllFiles('~/storage/data/aflw_cropped_context','.jpg');
        L_pts = load('~/storage/data/ptsData');
        ptsData = L_pts.ptsData(1:end);
        poses = L_pts.poses(1:end);
        requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
        all_kp_predictions_local = zeros(length(fra_db),length(requiredKeypoints),5);
        all_kp_predictions_global = zeros(length(fra_db),length(requiredKeypoints),5);
        %
        rolls = [poses.roll];
        pitches = [poses.pitch];
        yaws = [poses.yaw];
        goods = abs(rolls) < 30*pi/180;
        [u,iu] = sort(rolls,'descend');
        edges = [0 20 45 90];
        [b,ib] = histc(180*abs(yaws)/pi,edges);
        poseMap = [90 -90 30 -30 0 0];
        % load the dpm detections on aflw.
        dpmDetsPath = '~/storage/data/aflw_cropped_context/dpm_detections.mat';
        if (exist(dpmDetsPath,'file'))
            load(dpmDetsPath);
        else
            ress = zeros(length(paths),6);
            id = ticStatus( 'loading paths', .5);
            for p = 1:length(paths)
                detPath = j2m('~/storage/aflw_faces_baw',paths{p});
                load(detPath);
                nBoxes = size(res.detections.boxes,1);
                if (nBoxes > 0)
                    ress(p,:) = res.detections.boxes(1,:);
                end
                tocStatus(id,p/length(paths));
            end
            save(dpmDetsPath,'ress');
        end
        scores = ress(:,end);
        %
        
        load ~/storage/misc/aflw_with_pts.mat
        if (0)
            bad_imgs = false(size(paths));
            id = ticStatus( 'cropping imgs', .5);
            ims = {};
            pts = {};
            inflateFactor = 1.3;
            for t = 1:length(paths)
                curBox = round(ress(t,:));
                % make sure all keypoints are inside face detection.
                
                boxToCheck = inflatebbox(curBox(1:4),1.3,'both',false);
                nOutOfBox = ~inBox( boxToCheck, ptsData(t).pts);
                nOutOfBox = nnz(nOutOfBox)/length(nOutOfBox);
                if (nOutOfBox > .8)
                    t
                    clf; imagesc2(imread(paths{t})); plotBoxes(boxToCheck,'Color','r','LineWidth',2);
                    plotPolygons(ptsData(t).pts,'g.');
                    drawnow
                    %         pause
                    bad_imgs(t) = true;
                    continue
                end
                %     continue
                curBox(1:4) = round(inflatebbox(curBox(1:4),1.3,'both',false));
                ims{t} = cropper(imread(paths{t}),curBox);
                curPts = ptsData(t);
                curPts.pts = curPts.pts-repmat(curBox(1:2),size(curPts.pts,1),1);
                curPose = poses(t);
                pts{t} = curPts;
                if (mod(t,20)==0)
                    clf; imagesc2(ims{t});
                    plotPolygons(curPts.pts,'g.');
                    drawnow;
                end
                tocStatus(id,t/length(paths));
            end
            
            ims = ims(~bad_imgs);
            pts = pts(~bad_imgs);
            pts = [pts{:}];
            poses = poses(~bad_imgs);
            scores = scores(~bad_imgs);
            resizeFactors = cellfun(@(x) min(1,128/size(x,1)),ims);
            for u = 1:length(ims)
                ims{u} = imResample(ims{u},resizeFactors(u),'bilinear');
            end
            for u = 1:length(ims)
                pts(u).pts = pts(u).pts*resizeFactors(u);
            end
            
            %         for u = 1:length(ims)
            %             clf; imagesc2(ims{u});
            %             plotPolygons(pts(u).pts,'r+');
            %             pause
            %             drawnow
            %         end
            %
            save ~/storage/misc/aflw_with_pts.mat ims pts poses scores inflateFactor resizeFactors
        end
        %
        T_score = 2.45; % minimal face detection score...
        im_subset = row(find(scores > T_score));
%         im_subset = vl_colsubset(im_subset,10000,'random');
        %
        curImgs = ims(im_subset);
        %%XX = extractHogsHelper(curImgs);
        zero_borders = false;
        XX = getImageStackHOG(curImgs,wSize,true,zero_borders );
        XX = cat(2,XX{:});
        save(myDataPath,'XX','wSize','curImgs', 'curYaws', 'curPitches', 'ress','ptsData','im_subset','-v7.3');
    end
    
    initialized2 = true;
    kdtree = vl_kdtreebuild(XX);
end

% try to make a prediction for each keypoint....

requiredKeypoints = {ptsData.pointNames};
requiredKeypoints = unique(requiredKeypoints);
all_kps = getKPCoordinates_2(pts(im_subset),requiredKeypoints)+1;

% 
% for t = 1:50:length(curImgs)
%     clf; imagesc2(curImgs{t});
%     plotPolygons(squeeze(all_kps(t,:,:))+1,'g+');
%     pause
% end
    

% all_kps = getKPCoordinates_2(pts(im_subset),requiredKeypoints);

all_poses = poses(im_subset);
all_yaws = [all_poses.yaw];

[u,iu] = sort(all_yaws);
% figure(1)
% displayImageSeries(conf,curImgs(iu(1:1:end)),-1)

clear fernsStruct;
heights = cellfun(@(x) size(x,1), curImgs);
for ikpType = 1:length(requiredKeypoints)
    ikpType
    cur_kps = squeeze(all_kps(:,ikpType,:));
    goods = ~any(isnan(cur_kps),2);    
    fernsStruct(ikpType) = makePointPredictor(XX(:,goods),cur_kps(goods,:)./repmat(heights(goods)',1,2));  
end

%% extract the relevant image patch for each of the required keypoints to make a better prediction: 
% apply again the predictors to the training data 
iSubType =4
[pred_x] = fernsRegApply( double(XX'), fernsStruct(iSubType).ferns_x);
[pred_y] = fernsRegApply( double(XX'), fernsStruct(iSubType).ferns_y);
subPatches = {};
%%
clear sub_ferns;
% iSubType = 4;
cur_kps = squeeze(all_kps(:,iSubType,:));
pts_in_box = zeros(size(cur_kps));
p = .3;
for t = 1:length(curImgs)
    t
    cur_prediction = [pred_x(t),pred_y(t)];
    real_point = cur_kps(t,:);
    sz = size(curImgs{t});
    curBox = [cur_prediction cur_prediction]*sz(1);
    curBox = round(inflatebbox(curBox,sz(1)*p,'both',true));
    curSub = cropper(curImgs{t},curBox);
    pt_in_box = real_point-curBox(1:2)+1;
    pts_in_box(t,:) = pt_in_box/size(curSub,1);
    subPatches{t} = curSub;
    % % % %     
    % % % %     clf;
    % % % %     subplot(1,2,1); imagesc2(curImgs{t}); plotBoxes(curBox);
    % % % %     subplot(1,2,2);
    % % % %     drawnow
    % % % %     imagesc2(curSub);
    % % % %     plotPolygons(pt_in_box,'g+');pause
    % % % %    
end
goods = ~any(isnan(pts_in_box),2);
subPatches = subPatches(goods);
xx_1 = getImageStackHOG(subPatches,40);
kdtree2 = vl_kdtreebuild(xx_1);
% subFerns = makePointPredictor(xx_1,pts_in_box(goods,:),5000);
cur_pts_in_box = pts_in_box(goods,:);

net = init_nn_network('imagenet-vgg-s.mat');
[x_17,x_19] = extractDNNFeats(curImgs,net);


%%

for u =201:1:length(curImgs)   
    t = u
      curImgData = fra_db(t);
    detections = curImgData.raw_faceDetections.boxes(1,:);
    conf.get_full_image = false;
    I_orig = getImage(conf,curImgData);
    faceBox = inflatebbox(detections(1:4),inflateFactor,'both',false);
    I = cropper(I_orig,round(faceBox));
    x = getImageStackHOG(I,wSize,true,zero_borders );    
%     clf; imagesc2(I);         
%     I = curImgs{u};    
    x = getImageStackHOG(I,wSize,true,zero_borders );
    preds = zeros(length(requiredKeypoints),2);
    figure(1)
    clf; 
    subplot(1,2,1)    
    imagesc2(I);
    for ikpType = 1:length(requiredKeypoints)
        [pred_x] = fernsRegApply( double(x'), fernsStruct(ikpType).ferns_x);
        [pred_y] = fernsRegApply( double(x'), fernsStruct(ikpType).ferns_y);
        pred_x = pred_x*size(I,1);
        pred_y = pred_y*size(I,1);
        preds(ikpType,:) = [pred_x,pred_y];       
    end
    
    plotPolygons(preds,'g+');
    showCoords(preds);
    
    
    myPred = preds(iSubType,:);            
    sz = size2(I);     
      subplot(1,2,2);
    for u = 1:15
        
        curBox = round(inflatebbox(myPred,sz(1)*p,'both',true));
        curSub = cropper(I,curBox);
        xx_2 = getImageStackHOG(curSub,40);
        [inds,dists] = vl_kdtreequery(kdtree2, xx_1,xx_2,'NUMNEIGHBORS',100,'MAXNUMCOMPARISONS',1000);
%         figure(2); clf; imagesc2(mImage(subPatches(inds)));
%         figure(1)
        nnPred = mean(cur_pts_in_box(inds(1:end),:),1);
        
        %     [pred_x] = fernsRegApply( double(xx_2'), subFerns(1).ferns_x);
        %     [pred_y] = fernsRegApply( double(xx_2'), subFerns(1).ferns_y);
        sz2 = size2(curSub);
        %     pred_x = pred_x*sz2(1);
        %     pred_y = pred_y*sz2(1);
%         plotBoxes(curBox);
      
        imagesc2(curSub);
        
        %     plot(pred_x,pred_y,'g+');
        nnPred = nnPred*sz2(1);
        plotPolygons(nnPred,'m*');        
        myPred = nnPred+curBox(1:2)-1;
%         subplot(1,2,1); plotPolygons(myPred,'r+');
        drawnow
        pause(.01)
    end
    pause
end

% [ferns,ysPr] = fernsRegTrain(double(XX'),all_yaws(:),'loss','L2','eta',.4,...
%      'thrr',[0 .4],'reg',0.1,'S',2,'M',1000,'R',3,'verbose',1);
%%
 for t = 1:length(fra_db)
    t
    curImgData = fra_db(t);
    detections = curImgData.raw_faceDetections.boxes(1,:);
    conf.get_full_image = false;
    I_orig = getImage(conf,curImgData);
    faceBox = inflatebbox(detections(1:4),inflateFactor,'both',false);
    I = cropper(I_orig,round(faceBox));
    x = getImageStackHOG(I,wSize,true,zero_borders );
    
    clf; imagesc2(I); 
        
    preds = zeros(length(requiredKeypoints),2);
    
    for ikpType = 1:length(requiredKeypoints)
        [pred_x] = fernsRegApply( double(x'), fernsStruct(ikpType).ferns_x);
        [pred_y] = fernsRegApply( double(x'), fernsStruct(ikpType).ferns_y);
        pred_x = pred_x*size(I,1);
        pred_y = pred_y*size(I,1);
        preds(ikpType,:) = [pred_x,pred_y];        
    end
    
    plotPolygons(preds,'g+');
    showCoords(preds);
    
    %title(num2str(180*ys/pi));
    
    pause
 end
%%
for t = 1:length(curImgs)
    clf; imagesc2(curImgs{t});
    plotPolygons(squeeze(all_kps(t,:,:)),'g+')
    pause
    drawnow
end


% train a forest to predict the pose....

%%
debug_ = true;
if debug_
%     figure(1);
end
%
outDir = '~/storage/all_kp_preds_new';
ensuredir(outDir);

kpParams.debug_ = debug_;
kpParams.wSize = wSize;
% kpParams.extractHogsHelper = extractHogsHelper;
kpParams.im_subset = im_subset;
kpParams.requiredKeypoints = requiredKeypoints;

all_names = {pts.pointNames};
all_names = cat(1,all_names{:});

% all_names = unique(all_names);
%%


% make multiple keypoint detectors using the sift features
subset = false(size(all_kps,1),1);
subset(1:1:end) = true;

annParams.maxFeats = inf;
annParams.cutoff = inf;

images_w = cellfun2(@(x) imResample(x,[wSize wSize],'bilinear'), curImgs);
kps = all_kps;
for iImg = 1:size(kps,1)
    kps(iImg,:,:) = wSize(1)*kps(iImg,:,:)/size(curImgs{iImg},1);
end
annPredictors = struct('features',{},'offsets',{},'kdtree',{});

for iPred = 1:length(requiredKeypoints)
    iPred
    cur_kps = squeeze(kps(:,iPred,:))+1;
    goods = ~any(isnan(cur_kps),2) & subset & all(cur_kps < wSize & cur_kps > 1,2);
    annParams.maxFeats = 200;
    annParams.cutoff = inf;
    [annPredictors(iPred).features,annPredictors(iPred).offsets] = prepareANNStarData(conf,images_w(goods),squeeze(cur_kps(goods,:,:)),annParams);
    annPredictors(iPred).kdtree = vl_kdtreebuild(annPredictors(iPred).features,'Distance','L2');
end

%%
kpParams.knn = 100;
kpParams.requiredKeypoints = requiredKeypoints;
% load ~/code/3rdparty/dpm_baseline.matw
for t = 801:length(fra_db)
    t
    curImgData = fra_db(t);
    detections = curImgData.raw_faceDetections.boxes(1,:);
    conf.get_full_image = false;
    I_orig = getImage(conf,curImgData);
    faceBox = inflatebbox(detections(1:4),inflateFactor,'both',false);
    I = cropper(I_orig,round(faceBox));
    
    %     [ds, bs] = imgdetect(I, model,-.1);
%     x2(I_orig);
    
    %     clf; imagesc(I); pause;continue
    %     [kp_global] = myFindFacialKeyPoints_2(conf,I,bb,XX,kdtree,my_curImgs,ress,ptsData,kpParams,net);
    net = [];
    [kp_global] = myFindFacialKeyPoints_3(conf,I,XX,kdtree,curImgs,pts(im_subset),kpParams,[],annPredictors);
    %     for p = 1:5
    %         [kp_global,deformed] = myFindFacialKeyPoints_3(conf,deformed,XX,kdtree,curImgs,pts(im_subset),kpParams,[]);
    %     end
end

x2(curImgs{6479})
plotPolygons(pts(im_subset(6479)).pts,'g+')

%% load all deep nets features for these images...
aflw_feats_dir = '~/storage/aflw_face_deep_features';
all_feats = struct('feats_s',{},'feats_deep',{});
for u = 1:length(curImgs)
    if (mod(u,100)==0)
        disp(u/length(curImgs))
    end    
    all_feats(u) = load(fullfile(aflw_feats_dir,[num2str(u) '.mat']));
end

save ~/storage/misc/aflw_all_deep_feats.mat all_feats

% get fc6 features from vgg-s
feats_fc6 = {};
for t = 1:length(all_feats)
    feats_fc6{t} = all_feats(t).feats_s(1).x;
end
feats_fc6 = cat(2,feats_fc6{:});
% feats_fc6 = normalize_vec(feats_fc6);
% means = mean(feats_fc6,2);
% feats_fc6 = bsxfun(@minus,feats_fc6,means);
% learn to predict the different keypoints...
clear fernsStruct;
heights = col(cellfun(@(x) size(x,1), curImgs));

models = struct('Bx',{},'By',{});
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');

% make a toy example.....
x = rand(100,1);
y = 2*x+3+rand(size(x))*.1;
plot(x,y,'.')

my_toy_model = train(y,sparse(x'), 't 0 -s 11 -B 0','col');


%%
% try all at ones
for ikpType = 1:length(requiredKeypoints)
    ikpType
    cur_kps = squeeze(all_kps(:,ikpType,:))+1;
    goods = ~any(isnan(cur_kps),2);
%     BB = feats_fc6(:,goods)*feats_fc6(:,goods)';
%     lambda = .1;
%     B_lambda = (eye(size(BB,1))*lambda+BB)\feats_fc6(:,goods);        
%     models(ikpType).Bx = B_lambda*(cur_kps(goods,1)./heights(goods)');            
%     models(ikpType).wx = mean((cur_kps(goods,1)./heights(goods)'));
    
%     figure,plot(models(ikpType).Bx'*feats_fc6 - 0*models(ikpType).wx)
%     figure,plot((cur_kps(goods,1)./heights(goods)'))
    
%     models(ikpType).By = B_lambda*(cur_kps(goods,2)./heights(goods)');
%     models(ikpType).wy = mean((cur_kps(goods,2)./heights(goods)'));
%     fitlm(double(feats_fc6(:,goods))',cur_kps(goods,2)./heights(goods)');
    
%     models(ikpType).Bx = fitlm(double(feats_fc6(:,goods))',cur_kps(goods,1)./heights(goods)');
%     models(ikpType).By = fitlm(double(feats_fc6(:,goods))',cur_kps(goods,2)./heights(goods)');
%         
%     models(ikpType).Bx = regress(cur_kps(goods,1)./heights(goods)',double(feats_fc6(:,goods))');
%     models(ikpType).By = regress(cur_kps(goods,2)./heights(goods)',double(feats_fc6(:,goods))');
            
   % fernsStruct(ikpType) = makePointPredictor(feats_fc6(:,goods),cur_kps(goods,:)./repmat(heights(goods)',1,2),1500,2);
    models(ikpType).Bx = train(cur_kps(goods,1)./heights(goods),sparse(double(feats_fc6(:,goods))), 't 0 -s 11 -B 0','col');
    models(ikpType).By = train(rand(sum(goods),1)+(cur_kps(goods,2)./heights(goods)),sparse(double(feats_fc6(:,goods))), 't 0 -s 11 -B 0','col');
end



% model_x = svmtrain(double(kp_x(goods)), double(xx(:,goods)'), 't 0 -s 3');
% model_y = svmtrain(double(kp_y(goods)), double(xx(:,goods)'), 't 0 -s 3');
% 
% 
% all_kp_preds = zeros(size(all_kps));
% for ikpType = 1:length(requiredKeypoints)
%     ikpType
%     [pred_x] = fernsRegApply( double(feats_fc6'), fernsStruct(ikpType).ferns_x);
%     [pred_y] = fernsRegApply( double(feats_fc6'), fernsStruct(ikpType).ferns_y);
%     all_kp_preds(:,ikpType,1) = pred_x;
%     all_kp_preds(:,ikpType,2) = pred_y;
% end
%
net_s = init_nn_network('imagenet-vgg-s.mat');
% net_deep = init_nn_network('imagenet-vgg-verydeep-16.mat');
%%
figure(1)
for t = 1:length(fra_db)
        
    curImgData = fra_db(t);
    detections = curImgData.raw_faceDetections.boxes(1,:);
    conf.get_full_image = false;
    I_orig = im2uint8(getImage(conf,curImgData));
    faceBox = inflatebbox(detections(1:4),inflateFactor,'both',false);
    I = cropper(I_orig,round(faceBox));
        
    feats_s = extractDNNFeats(I,net_s,[16]); % get fc6,fc7,fc8 for both nets.
%     feats_deep = extractDNNFeats(I,net_deep,33); % get fc6,fc7,fc8 for both nets.
    curFeats = feats_s(1).x;
    
    %I = subImgs{t};
%     if (isempty(subImgs{t}))
%         continue
%     end
    %L = load(fullfile('~/storage/fra_db_face_deep_features',[num2str(t) '.mat']));
    %curFeats = L.feats_s(1).x;
%     I = curImgs{t};
    h = size(I,1);
    clf; imagesc2(I);
    curFeats = sparse(double(normalize_vec(curFeats)));
%     curFeats = curFeats-means;
    for ikpType = 1%:length(requiredKeypoints)
        [~,~,pred_x] = predict(0, curFeats,models(ikpType).Bx,'','col');
        [~,~,pred_y] = predict(0, curFeats,models(ikpType).By,'','col');
%                 pred_x = predict(models(ikpType).Bx,curFeats');
%                 pred_y = predict(models(ikpType).By,curFeats');
        
%         pred_y = models(ikpType).By'*curFeats+models(ikpType).wy;        
%         pred_x = models(ikpType).Bx'*curFeats+models(ikpType).wx;
%         pred_y = models(ikpType).By'*curFeats+models(ikpType).wy;
%         fitlm
        %[pred_x] = fernsRegApply( double(curFeats'), fernsStruct(ikpType).ferns_x)
        %[pred_y] = fernsRegApply( double(curFeats'), fernsStruct(ikpType).ferns_y)
        plot(pred_x*h,pred_y*h,'g+');
    end        
%     cur_kps = squeeze(all_kps(t,:,:));
%     plotPolygons(1+cur_kps,'m*');
    drawnow
    pause
end



% feats_fc6 = normalize_vec(feats_fc6);


% showNN(feats_fc6(:,1:5:end),curImgs(1:5:end));
