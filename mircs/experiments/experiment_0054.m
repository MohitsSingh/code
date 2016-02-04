%%% experiment_0054 - predict keypoints using regression from fc6 features.
%% 28/12/2014
% Make my own facial landmark detector.
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    %wSize = 96;
    wSize = 64;
    extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[wSize wSize],'bilinear')))) , y);
    initialized = true;
    requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    addpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7'));
        
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    initialized = true;
    conf.get_full_image = true;
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
    load ~/storage/mircs_18_11_2014/s40_fra
    params.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    load('~/storage/mircs_18_11_2014/allPtsNames','allPtsNames');
    [~,~,reqKeyPointInds] = intersect(params.requiredKeypoints,allPtsNames);
    
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
    load ~/storage/mircs_18_11_2014/s40_fra
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
    
    default_params = defaultPipelineParams(true);

end
%%
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
        bad_imgs = false(size(paths));
        id = ticStatus( 'cropping imgs', .5);
        ims = {};
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
            ims{t} = cropper(imread(paths{t}),curBox);
            tocStatus(id,t/length(paths));
        end
        
        %
        scores = ress(~bad_imgs,end);
        ims = ims(~bad_imgs);
        ptsData = ptsData(~bad_imgs);
        ress(bad_imgs,:) = [];
        yaws(bad_imgs) = [];
        pitches(bad_imgs) = [];
        goodPaths = paths;goodPaths(bad_imgs) = [];
        %
        T_score = 2.45; % minimal face detection score...
        im_subset = row(find(scores > T_score));
        im_subset = vl_colsubset(im_subset,10000,'random');
        %
        curImgs = ims(im_subset);
        curYaws = 180*yaws(im_subset)/pi;
        curPitches = 180*pitches(im_subset)/pi;
        XX = extractHogsHelper(curImgs);
        XX = cat(2,XX{:});
        save(myDataPath,'XX','wSize','curImgs', 'curYaws', 'curPitches', 'ress','ptsData','im_subset','bad_imgs','-v7.3');
    end
    
    initialized2 = true;
    kdtree = vl_kdtreebuild(XX);
end
%%
debug_ = true;
if debug_
    figure(1);
end

%%
%%  trying with neural net features....

% run the CNN
% kp_params_orig = kpParams;
% kpParams = kp_params_orig;
debug_ = false;
kpParams.debug_ = debug_;
kpParams.wSize = wSize;
kpParams.extractHogsHelper = extractHogsHelper;
kpParams.im_subset = im_subset;
kpParams.requiredKeypoints = requiredKeypoints;
% yaws = yaws(im_subset);
sel_ = vl_colsubset(1:length(im_subset),5000,'uniform');
sel_val = sel_-1;
my_yaws = yaws(sel_);
% sel_ = 1:length(curImgs);
kpParams.im_subset = kpParams.im_subset(sel_);
my_curImgs = curImgs(sel_);

% imo = cnn_imagenet_get_batch(my_curImgs, 'averageImage',net.normalization.averageImage,...
%     'border',net.normalization.border,'keepAspect',net.normalization.keepAspect,...
%     'numThreads', 1, ...
%     'prefetch', false,...
%     'augmentation', 'none','imageSize',net.normalization.imageSize);
[x_17,x_19] = extractDNNFeats(my_curImgs,net);

% visualize some of the keypoints on this dataset for sanity
%[cofw_res(t).kp_global,cofw_res(t).kp_local] = myFindFacialKeyPoints_new(conf,I,bb,XX,kdtree,curImgs,ress,ptsData,kpParams,all_offsets_n,kdtree2,all_feats_n);
my_rects = ress(im_subset(sel_),:);
kps = getKPCoordinates(ptsData(kpParams.im_subset),my_rects,requiredKeypoints);

img_heights = cellfun(@(x) size(x,1),my_curImgs);

% get the first kp pair....
xx = normalize_vec(x_17);
iKP = 3;
kp_x = squeeze(kps(:,iKP,1))./img_heights(:);
kp_y = squeeze(kps(:,iKP,2))./img_heights(:);
goods = ~isnan(kp_x) & ~isnan(kp_y);

model_x = svmtrain(double(kp_x(goods)), double(xx(:,goods)'), 't 0 -s 3');
model_y = svmtrain(double(kp_y(goods)), double(xx(:,goods)'), 't 0 -s 3');

[u,iu] = sort(my_yaws);
displayImageSeries(conf,my_curImgs(iu),0.1)

% predict the yaw

[ferns_x,xPr] = fernsRegTrain(double(xx(:,goods)'),double(kp_x(goods)),'loss','L2','eta',1,...
   'thrr',[-.1 .1],'reg',.1,'S',8,'M',50,'R',10,'verbose',1);
[ferns_y,yPr] = fernsRegTrain(double(xx(:,goods)'),double(kp_y(goods)),'loss','L2','eta',1,...
   'thrr',[-.1 .1],'reg',.1,'S',8,'M',50,'R',10,'verbose',1);

%% sub-image prediction
sub_rects = inflatebbox([kp_x,kp_y,kp_x,kp_y],[img_heights/4],'both',true);
sub_rects = zeros(size(kp_x,1),4);

kp_xy = [kp_x kp_y];
mm = mean(kp_xy(goods,:));
kp_xy(~goods,1) = mm(1);kp_xy(~goods,2) = mm(2);

for t = 1:size(sub_rects,1);
    sub_rects(t,:) = round(inflatebbox([kp_xy(t,:) kp_xy(t,:)]*img_heights(t),[img_heights(t)/3],'both',true));
end

mySubImgs = {};

for t = 1:length(kp_x)
    mySubImgs{t} = cropper(my_curImgs{t},sub_rects(t,:));
end

[a,b,c] = BoxSize(sub_rects);

%%
figure(1)
for t = 1:length(my_curImgs);
    curImg = my_curImgs{t};
    clf; imagesc2(curImg);
%     plotPolygons(squeeze(kps(t,:,:)),'g+');
    h = size(curImg,1);
    plot(kp_x(t)*h,kp_y(t)*h,'g*');
%     plotBoxes(my_rects(t,:)
    drawnow
    pause
end

%%

for t = 800:length(fra_db)
    conf.get_full_image = false;
    I_orig = getImage(conf,fra_db(t));
    curBB = round(fra_db(t).raw_faceDetections(1).boxes(1,:));
    I = cropper(I_orig,curBB);  
    clf;imagesc2(I);
    x_17_cur = extractDNNFeats(I,net);
    x_17_cur = normalize_vec(x_17_cur);
    x_pred = fernsRegApply(double(x_17_cur)',ferns_x)*size(I,1);
    y_pred = fernsRegApply(double(x_17_cur)',ferns_y)*size(I,1);    
    plot(x_pred,y_pred,'g+');
    pause    
end

%%
imgs_val = curImgs(sel_val(sel_val > 1));

for t = 1:length(imgs_val)
    I = imgs_val{t};    
    x_17_cur = extractDNNFeats(I,net);
     x_17_cur = normalize_vec(x_17_cur);
    x_pred = fernsRegApply(double(x_17_cur)',ferns_x)*size(I,1);
    y_pred = fernsRegApply(double(x_17_cur)',ferns_y)*size(I,1);    
    clf; imagesc2(I);
    plot(x_pred,y_pred,'g+');
    pause    
end

