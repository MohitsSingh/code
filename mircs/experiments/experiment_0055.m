%% Experiment 0055 %%%%%
%% 8/9/2014
% Make my own facial landmark detector.
if (~exist('initialized','var'))
    initpath
    config
    requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    myDataPath = '~/storage/misc/kp_pred_data.mat';
    addpath('/home/amirro/code/3rdparty/uri/');
    load(myDataPath);
    dpmDetsPath = '~/storage/data/aflw_cropped_context/dpm_detections.mat';
    load(dpmDetsPath);
    load ~/storage/misc/aflw_with_pts.mat %ims pts poses scores inflateFactor resizeFactors
    %     im_subset = 1:length(scores);
    %     im_subset = 1:length(scores);
    %%%T_score = 2.45; % minimal face detection score...
    scores = ress(:,end);
    T_score = 3.45; % minimal face detection score...
    
    %im_subset = row(find(scores > T_score));
    
    %     im_subset = vl_colsubset(im_subset,10000,'uniform');
    %
    curImgs = ims(im_subset);
    %%XX = extractHogsHelper(curImgs);
    zero_borders = false;wSize = 48;
    XX_yaw = getImageStackHOG(curImgs,wSize,true,zero_borders );
    R = load('~/storage/misc/all_face_data.mat');
    %     poses = [all_face_data.pose];
    %     save(myDataPath,'XX','wSize','curImgs', 'curYaws', 'curPitches', 'ress','ptsData','im_subset','-v7.3');
    
    
    % split into different directions according to ress
    cur_ress = ress(im_subset,:);
    
end
%%
% [paths,names] = getAllFiles('~/storage/data/aflw_cropped_context','.jpg');
% load ~/storage/misc/all_face_data.mat
% U = load('~/storage/misc/aflw_all_deep_feats.mat');
close all
% displayImageSeries(conf,paths(1:10:end),.001)
% im_subset = 1:length(ims
% im_subset = 1:length(ims);
all_poses = poses(im_subset);
all_yaws = [all_poses.yaw];
all_rolls = [all_poses.roll];
all_pitches= [all_poses.pitch];
[u,iu] = sort(all_yaws,'descend');
% sel_ = abs(all_yaws) >= 0.78;
% cur_rolls = all_rolls(sel_);
displayImageSeries_simple(curImgs(iu(1:10:end)),0.01)
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');

for t = 1:50:length(all_yaws)
    clf; imagesc2(curImgs{iu(t)}); title(num2str(180*u(t)/pi));    
    pause
end

yaws_det = cur_ress(:,5);
t1 = curImgs(yaws_det==1);
% cur_ress(yaws_det==1,end)

% 90 : left, 0 : front -90: right

yaw_model = train(double(all_yaws(1:end))',sparse(double(XX_yaw(:,1:end))), '-c .1 -s 11 -B 0','col');
save yaw_model yaw_model

% roll_model = train(double(all_rolls(1:end))',sparse(double(XX(:,1:end))), '-c 1 -s 11 -B -1','col');
a1 = yaw_model.w(1:end-1)*XX_yaw(:,1:1:end);

q = 4305
a1(q)
all_yaws(q)


subset = 1:50:length(curImgs);

imgs1 = curImgs(subset);
yaws1 = all_yaws(subset);
[ff,iff] = sort(yaws1,'descend');
x2(imgs1(iff));
net = init_nn_network();
[res] = extractDNNFeats(imgs1,net,16);

labels1 = find(all_yaws < -.8);
labels2_1 = find(all_yaws > - .5);
labels2_2 = find(all_yaws < .5);
labels3 = find(all_yaws > .8);

face_right_labels = [ones(size(labels1)),-ones(size(labels2))];
xx_faces_right = XX_yaw

yaw_model1 = train(double(all_yaws(1:end))',sparse(double(XX_yaw(:,1:end))), '-c .001 -s 11 -B 0','col');


x2(curImgs(labels1(1:10:end)));

% train
% there is something very strange going on here...


mImage(curImgs(abs(all_yaws+.4) < .1));
differences = abs(a1-all_yaws(1:1:end)); 
[a,b] = hist(abs(a1-all_yaws(1:1:end)),51)
figure(1); plot(180*b/pi,cumsum(a)/sum(a)); title('hog');
[r,ir] = sort(differences,'descend'); r(1)
% for q = 1:length(r)
%     k = ir(q);
%     clf; imagesc2(curImgs{k});
%     predicted_yaw = a1(k)
%     real_yaw = all_yaws(k)    


figure,bar(b,a)
load ~/storage/misc/s40_fra_faces_d_new
fra_db = s40_fra_faces_d
%%
%load ~/storage/misc/s40_face_detections.mat; % all_detections

% load ~/storage/misc/s40_face_dets.mat; % all_detections

imgs = {};
scores = {};
for t = 1:30:4000
    t
    curImg = s40_fra_faces_d(t);
    conf.get_full_image = false;
    assert(strcmp(all_detections(t).imageID,curImg.imageID));
    faceBox = all_detections(t).detections.boxes(1,:);
    curScore = faceBox(end);
    if (curScore<.2),continue,end
    faceBox = faceBox(1:4);
    I = getImage(conf,curImg);
    %     faceBox = curImg.faceBox;
    %faceBox = curImg.raw_faceDetections.boxes(1,:);
    faceBox = round(inflatebbox(faceBox,1.3,'both',false));
    I_face = cropper(I,faceBox);
    imgs{end+1} = I_face;
    scores{end+1} = curScore;
end

%%
cur_image_hog = getImageStackHOG(imgs,wSize,true,zero_borders );
yaw_prediction = yaw_model.w(1:end-1)*cur_image_hog;
showSorted(imgs,yaw_prediction);


%% use the rcpr results...
addpath('/home/amirro/code/3rdparty/rcpr_v1');
load regModel_frontal
load regModel_profile
T=100;K=15;L=20;RT1=5;
ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);
prm=struct('thrr',[-1 1]/5,'reg',.01);
occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',3,'th',.5);
regPrm_frontal = struct('type',1,'K',K,'occlPrm',occlPrm,...
    'loss','L2','R',0,'M',5,'model',regModel_frontal.model,'prm',prm);
regPrm_profile = struct('type',1,'K',K,'occlPrm',occlPrm,...
    'loss','L2','R',0,'M',5,'model',regModel_profile.model,'prm',prm);
prunePrm=struct('prune',1,'maxIter',2,'th',0.15,'tIni',10);
% I = getImage(conf,imgData.imageID);
% I = cropper(I,imgData.raw_faceDetections);
%fra_db = s40_fra_faces_d;
%%
%%
load ~/storage/misc/landmark_training_data %faces_and_landmarks_frontal faces_and_landmarks_profile
%%
zero_borders = true;
wSize = 48;
Is_frontal = {faces_and_landmarks_frontal.I};
nFrontal = length(Is_frontal);
XX_frontal = getImageStackHOG(Is_frontal,wSize,true,zero_borders );
Is_profile = {faces_and_landmarks_profile.I};
XX_profile = getImageStackHOG(Is_profile,wSize,true,zero_borders );
II = [Is_frontal,Is_profile];
XX = [XX_frontal XX_profile];
kdtree = vl_kdtreebuild(XX,'Distance','L1');
face_comps = ones(size(II));
face_comps(length(Is_frontal)+1:end) = 2;
% all_c = zeros(size(face_comps));
all_c = [[faces_and_landmarks_frontal.c],[faces_and_landmarks_profile.c]];
% all_c(length(Is_frontal)+1:end) = [faces_and_landmarks_profile.c];
% mImage(Is_profile(my_inds-length(Is_frontal)+1))

%%
close  all
% fra_db = s40_fra_faces_d
RT1 = 30;
% initialize
% 810 : indian girl with white cup
poseMap = [90 -90 30 -30 0 0];
% nn_initialization
for iImg =  1:5:length(s40_fra_faces_d)
    iImg
    for u = 1
        %for nn_initialization = [false true]
        for nn_initialization = [true]
            curImgData = fra_db(iImg);
            if (curImgData.classID ~= conf.class_enum.DRINKING)
                continue
            end
            conf.get_full_image = true;
            [I_orig,I_rect] = getImage(conf,curImgData);
            if (~curImgData.isTrain)
                %faceBox = all_detections(iImg).detections.boxes(1,:);
                faceBox = s40_person_face_dets(iImg).boxes_rot(1,:);
                curPose = poseMap(faceBox(5));
            end
% % %             else
% % %                 'training'
% % %                 curFaceBoxes = all_detections(iImg).detections.boxes(:,1:4);
% % %                 %                 x2(I_orig);
% % %                 %                 plotBoxes(curFaceBoxes);
% % %                 real_face_box = curImgData.faceBox-I_rect([1 2 1 2]);
% % %                 %                 plotBoxes(real_face_box,'r--','LineWidth',2);
% % %                 ovps = boxesOverlap(real_face_box,curFaceBoxes);
% % %                 [~,iz] = max(ovps);
% % %                 faceBox = all_detections(iImg).detections.boxes(iz,:);
% % %                 curPose = poseMap(faceBox(5));
% % %             end
            fprintf('comp: %d,pose: %d\n',faceBox(5),curPose);
            curScore = faceBox(6);
            curRot = faceBox(7);            
            I = cropper(imrotate(I_orig,curRot,'bilinear','crop'),round(faceBox));
            %             clf; imagesc2(I); pause; continue
            
            %         q = .03;
            %         dq = .01;
            %         range = -q:dq:q;
            %         for dx = range
            %             for dy = range
            %         face_box_1 = inflatebbox(faceBox,1.3,'both',false);
            % %                 s = face_box_1(3)-face_box_1(1);
            %                 face_box_1([1 3])=face_box_1([1 3])+dx*s;
            %                 face_box_1([2 4])=face_box_1([2 4])+dy*s;
            face_box_1 = inflatebbox(faceBox,1.3,'both',false);
            I_rots = {};
            %         clf;imagesc(I); pause;continue
            rotations = -15:5:15;
            %                     rotations = 0;
            
            zooms = 1;%[.8 1 1.1]
            for iZoom = 1:length(zooms)
                for iRot = 1:length(rotations)
                    curRot = rotations(iRot);
                    I_1 = cropper(I_orig,round(inflatebbox(face_box_1,zooms(iZoom),'both',false)));
                    I_1 = imrotate(I_1,curRot,'bilinear','crop');
                    I_rots{end+1} = clip_to_bounds(I_1);
                end
            end
            %         roll_prediction= 180*roll_model.w*cur_image_hog/pi;
            %         fprintf('roll prediction : %f\n', roll_prediction);
            %         I_1 = imrotate(I_1,-roll_prediction,'bilinear','crop');
            cur_image_hog = getImageStackHOG(I_rots ,wSize,true,zero_borders);
            [ind_all,dist_all] = vl_kdtreequery(kdtree,XX,cur_image_hog,'numneighbors',RT1,'MaxNumComparisons',1000);
            ind_all = double(ind_all);
            [mm,imm] = min(dist_all(1,:));
            best_rotation = rotations(imm);
            I = imrotate(I,best_rotation,'bilinear','crop');
            figure(1);clf; subplot(1,2,1); imagesc2(I_rots{imm});
            %             subplot(1,2,2); imagesc2(mImage(II(ind_all(:,imm))));
            comps = face_comps(ind_all(:,imm));
            is_right = ~mode(double(all_c(ind_all(1:RT1,imm))>=11));
            all_c(ind_all(1:RT1,imm));
            %if (~is_left),continue,end
            cur_comp = mode(comps(1:5));
            %%%%frontal = cur_comp==1;
            
            frontal = abs(curPose)~=90;
            % %             if frontal,continue,end
            
            %         pause;
            %         continue
            %         end
            %         continue
            %             end
            % %         end
            %         continue
            %         yaw_prediction = 180*yaw_model.w(1:end-1)*cur_image_hog/pi;
            %         fprintf('yaw prediction : %f\n', yaw_prediction);
            %         I = imResample(I,1,'bilinear');
            %         frontal = abs(yaw_prediction)<30;
            if (~frontal && curPose==90) % && yaw_prediction > 0)
                curPose
                I =  flip_image(I);
                %                 pause
                %         else
                %             I = flip_image(I);
            end
            
            if (curPose==30)
                I =  flip_image(I);
            end
            
            beVerbose = 0;
            ff = [1 1 fliplr(size2(I)-1)];
            RT1_1 = RT1;
            my_inds = [];
            if (frontal)
                if nn_initialization
                    %                     my_inds = ind_all(comps==1,imm);
                    
                    my_inds = find(all_c>3);
                    my_inds = vl_colsubset(my_inds,RT1,'random');
                    
                    %                 my_inds = my_inds(1:2)
                    RT1_1 = min(RT1,length(my_inds));
                end
                %             my_inds =
                p=shapeGt('initTest',{I},ff,regModel_frontal.model,...
                    regModel_frontal.pStar,regModel_frontal.pGtN,RT1_1,my_inds);
                testPrm = struct('RT1',RT1_1,'pInit',ff,...
                    'regPrm',regPrm_frontal,'initData',p,'prunePrm',prunePrm,...
                    'verbose',beVerbose);
                t=clock;[p,pRT] = rcprTest({I},regModel_frontal,testPrm);t=etime(clock,t);
            else
                if nn_initialization
                    %                     my_inds = ind_all(comps==2,imm)-length(Is_frontal);
                    
                    my_inds = find(all_c<=3)-nFrontal;
                    my_inds = vl_colsubset(my_inds,RT1,'random');
                    %                 my_inds = my_inds(1:2)
                    RT1_1 = min(RT1,length(my_inds));
                end
                %             RT1_1 = min(RT1,length(my_inds));
                p_init=shapeGt('initTest',{I},ff,regModel_profile.model,...
                    regModel_profile.pStar,regModel_profile.pGtN,RT1_1,my_inds);
                testPrm = struct('RT1',RT1_1,'pInit',ff,...
                    'regPrm',regPrm_profile,'initData',p_init,'prunePrm',prunePrm,...
                    'verbose',beVerbose);
                t=clock;[p,pRT] = rcprTest({I},regModel_profile,testPrm);t=etime(clock,t);
            end
            figure(1);clf
            vl_tightsubplot(1,2,1);imagesc2(clip_to_bounds(I));
            vl_tightsubplot(1,2,2);imagesc2(clip_to_bounds(I));
            plot(p(1:end/2),p(end/2+1:end),'gd','LineWidth',2);
            ensuredir('~/notes/images/2015_2_10');
            saveas(gcf,['~/notes/images/2015_2_10/' curImgData.imageID(1:end-4) '.png']);
            %              disp(['nn_initialization: ',num2str(nn_initialization)])
            pause
            
        end
    end
end


%%
%
%%
myDataPath = '~/storage/misc/kp_pred_data_new.mat';
bad_imgs = false(size(paths));
id = ticStatus( 'cropping imgs', .5);
ims = {};
pts = {};
for t = 1:length(paths)
    curBox = round(ress(t,:));
    % make sure all keypoints are inside face detection.    
    %boxToCheck = inflatebbox(curBox(1:4),1.3,'both',false);
    boxToCheck=curBox(1:4);
    nOutOfBox = ~inBox( boxToCheck, ptsData(t).pts);
    nOutOfBox = nnz(nOutOfBox)/length(nOutOfBox);
    if (nOutOfBox > .2)
        t
        clf; imagesc2(imread(paths{t})); plotBoxes(boxToCheck,'Color','r','LineWidth',2);
        plotPolygons(ptsData(t).pts,'g.');
        drawnow
%                 pause
        bad_imgs(t) = true;
        continue
    end
    clf; imagesc2(imread(paths{t})); plotBoxes(boxToCheck,'Color','r','LineWidth',2);
    plotPolygons(ptsData(t).pts,'g+');
    drawnow
    pause
    %     continue
    ims{t} = cropper(imread(paths{t}),curBox);
    tocStatus(id,t/length(paths));
end

resampler = @(x) imResample(x,[48 48],'bilinear');
ims_small = cellfun2(resampler,ims);
save(myDataPath,'ims_small',  'ress','ptsData','bad_imgs','poses','-v6');
scores = ress(:,end);

load(myDataPath);
% retrain for the different components...
ovps = boxesOverlap(ress(:,1:4),[64 64 192 192]);
%sel_ = scores > 3 & ~bad_imgs(:) & ovps > .6;
sel_ = scores > 4 & ~bad_imgs(:) & ress(:,5)==1;
x2(ims_small(sel_));





ims_small = ims_small(sel_);
ress = ress(sel_,:);
ptsData = ptsData(sel_);
poses = poses(sel_);
yaws = [poses.yaw];
x2(ims_small(1:100:end));

%%
all_poses = poses;
all_yaws = [all_poses.yaw];
all_rolls = [all_poses.roll];
all_pitches= [all_poses.pitch];
[u,iu] = sort(all_yaws,'descend');
% [u,iu] = sort(all_rolls,'descend');
% sel_ = abs(all_yaws) >= 0.78;
% cur_rolls = all_rolls(sel_);
% displayImageSeries_simple(curImgs(iu(1:10:end)),0.01)
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
for t = 1:500:length(all_yaws)
    clf; imagesc2(ims_small{iu(t)}); title(num2str(180*u(t)/pi));    
    pause
end

zero_borders = false;wSize = 48;
XX_yaw = getImageStackHOG(ims_small,wSize,true,zero_borders );
yaw_model = train(double(all_yaws(1:end))',sparse(double(XX_yaw(:,1:end))), '-c 1 -s 11 -B 0','col');
save yaw_model yaw_model

a1 = yaw_model2.w(1:end-1)*XX_yaw(:,1:1:end);

differences = abs(a1-all_yaws(1:1:end)); 
[a,b] = hist(abs(a1-all_yaws(1:1:end)),51)
figure(1);clf; plot(180*b/pi,cumsum(a)/sum(a)); 
[r,ir] = sort(differences,'descend'); r(1)
for q = 1:length(r)
    k = ir(q);
    clf; imagesc2(ims_small{k});
    predicted_yaw = a1(k)
    real_yaw = all_yaws(k)    
    pause
end

% x2(ims_small(ress(:,5)==3));


% roll_model = train(double(all_rolls(1:end))',sparse(double(XX(:,1:end))), '-c 1 -s 11 -B -1','col');
a1 = yaw_model.w(1:end-1)*XX_yaw(:,1:1:end);
q = 4305
a1(q)
all_yaws(q)
subset = 1:50:length(curImgs);
imgs1 = curImgs(subset);
yaws1 = all_yaws(subset);
[ff,iff] = sort(yaws1,'descend');
x2(imgs1(iff));
net = init_nn_network();
[res] = extractDNNFeats(imgs1,net,16);

labels1 = find(all_yaws < -.8);
labels2_1 = find(all_yaws > - .5);
labels2_2 = find(all_yaws < .5);
labels3 = find(all_yaws > .8);


face_right_labels = [ones(size(labels1)),-ones(size(labels2))];
xx_faces_right = XX_yaw
yaw_model1 = train(double(all_yaws(1:end))',sparse(double(XX_yaw(:,1:end))), '-c .001 -s 11 -B 0','col');
x2(curImgs(labels1(1:10:end)));
% train
% there is something very strange going on here...

[ IDX, C, d ] = kmeans2( all_yaws', 5,'maxIter',1000,'display',true);%, varargin )
 
new_yaws = C(IDX);
figure,plot(all_yaws); hold on; plot(new_yaws,'r-');

f_sel = find(sel_);
figure,plot(all_yaws)
x2(ims_small(720))
[u,iu] = sort(all_yaws);
figure,plot(u);
hold on,plot(ress(iu,5),'r-')
f_sel(6229)
yaw_model2 = train(double(new_yaws(1:2:end)),sparse(double(XX_yaw(:,1:2:end))), '-c .01 -s 11 -B 0','col');
k = 13162
% poses(k)
a1 = 180*yaw_model2.w(1:end-1)*XX_yaw(:,1:1:end)/pi;
differences = abs(a1'-180*new_yaws(1:1:end)/pi); 
[a,b] = hist(differences,51);
figure(1); plot(b/pi,cumsum(a)/sum(a)); 
[r,ir] = sort(differences,'descend'); r(1)
% for q = 1:length(r)
%     k = ir(q);
%     clf; imagesc2(ims_small{k});
%     predicted_yaw = a1(k)
%     real_yaw = 180*all_yaws(k)/pi
%     pause
% end
%%
% % % close  all
% % % % fra_db = s40_fra_faces_d
% % % RT1 = 30;
% % % % initialize
% % % % 810 : indian girl with white cup
% % % poseMap = [90 -90 30 -30 0 0];
% % % % nn_initialization
% % % for iImg =  1:5:length(s40_fra_faces_d)
% % %     iImg
% % %     for u = 1
% % %         %for nn_initialization = [false true]
% % %         for nn_initialization = [true]
% % %             curImgData = fra_db(iImg);
% % %             if (curImgData.classID ~= conf.class_enum.DRINKING)
% % %                 continue
% % %             end
% % %             conf.get_full_image = true;
% % %             [I_orig,I_rect] = getImage(conf,curImgData);
% % %             if (~curImgData.isTrain)
% % %                 %faceBox = all_detections(iImg).detections.boxes(1,:);
% % %                 faceBox = s40_person_face_dets(iImg).boxes_rot(1,:);
% % %                 curPose = poseMap(faceBox(5));
% % %             else
% % %                 'training'
% % %                 curFaceBoxes = all_detections(iImg).detections.boxes(:,1:4);
% % %                 %                 x2(I_orig);
% % %                 %                 plotBoxes(curFaceBoxes);
% % %                 real_face_box = curImgData.faceBox-I_rect([1 2 1 2]);
% % %                 %                 plotBoxes(real_face_box,'r--','LineWidth',2);
% % %                 ovps = boxesOverlap(real_face_box,curFaceBoxes);
% % %                 [~,iz] = max(ovps);
% % %                 faceBox = all_detections(iImg).detections.boxes(iz,:);
% % %                 curPose = poseMap(faceBox(5));
% % %             end
% % %             fprintf('comp: %d,pose: %d\n',faceBox(5),curPose);
% % %             curScore = faceBox(end);
% % %                         if (curScore<.2),continue,end
% % %             faceBox = faceBox(1:4);
% % %             %         faceBox = detections(1:4);
% % %             faceBox = faceBox+I_rect([1 2 1 2])-1;
% % %             faceBox = inflatebbox(faceBox,1,'both',false);
% % %             I = cropper(I_orig,round(faceBox));
% % %             %             clf; imagesc2(I); pause; continue
% % %             
% % %             %         q = .03;
% % %             %         dq = .01;
% % %             %         range = -q:dq:q;
% % %             %         for dx = range
% % %             %             for dy = range
% % %             %         face_box_1 = inflatebbox(faceBox,1.3,'both',false);
% % %             % %                 s = face_box_1(3)-face_box_1(1);
% % %             %                 face_box_1([1 3])=face_box_1([1 3])+dx*s;
% % %             %                 face_box_1([2 4])=face_box_1([2 4])+dy*s;
% % %             face_box_1 = inflatebbox(faceBox,1.3,'both',false);
% % %             I_rots = {};
% % %             %         clf;imagesc(I); pause;continue
% % %             rotations = -15:5:15;
% % %             %                     rotations = 0;
% % %             
% % %             zooms = 1;%[.8 1 1.1]
% % %             for iZoom = 1:length(zooms)
% % %                 for iRot = 1:length(rotations)
% % %                     curRot = rotations(iRot);
% % %                     I_1 = cropper(I_orig,round(inflatebbox(face_box_1,zooms(iZoom),'both',false)));
% % %                     I_1 = imrotate(I_1,curRot,'bilinear','crop');
% % %                     I_rots{end+1} = clip_to_bounds(I_1);
% % %                 end
% % %             end
% % %             %         roll_prediction= 180*roll_model.w*cur_image_hog/pi;
% % %             %         fprintf('roll prediction : %f\n', roll_prediction);
% % %             %         I_1 = imrotate(I_1,-roll_prediction,'bilinear','crop');
% % %             cur_image_hog = getImageStackHOG(I_rots ,wSize,true,zero_borders);
% % %             [ind_all,dist_all] = vl_kdtreequery(kdtree,XX,cur_image_hog,'numneighbors',RT1,'MaxNumComparisons',1000);
% % %             ind_all = double(ind_all);
% % %             [mm,imm] = min(dist_all(1,:));
% % %             best_rotation = rotations(imm);
% % %             I = imrotate(I,best_rotation,'bilinear','crop');
% % %             figure(1);clf; subplot(1,2,1); imagesc2(I_rots{imm});
% % %             %             subplot(1,2,2); imagesc2(mImage(II(ind_all(:,imm))));
% % %             comps = face_comps(ind_all(:,imm));
% % %             is_right = ~mode(double(all_c(ind_all(1:RT1,imm))>=11));
% % %             all_c(ind_all(1:RT1,imm));
% % %             %if (~is_left),continue,end
% % %             cur_comp = mode(comps(1:5));
% % %             %%%%frontal = cur_comp==1;
% % %             
% % %             frontal = abs(curPose)~=90;
% % %             % %             if frontal,continue,end
% % %             
% % %             %         pause;
% % %             %         continue
% % %             %         end
% % %             %         continue
% % %             %             end
% % %             % %         end
% % %             %         continue
% % %             %         yaw_prediction = 180*yaw_model.w(1:end-1)*cur_image_hog/pi;
% % %             %         fprintf('yaw prediction : %f\n', yaw_prediction);
% % %             %         I = imResample(I,1,'bilinear');
% % %             %         frontal = abs(yaw_prediction)<30;
% % %             if (~frontal && curPose==90) % && yaw_prediction > 0)
% % %                 curPose
% % %                 I =  flip_image(I);
% % %                 %                 pause
% % %                 %         else
% % %                 %             I = flip_image(I);
% % %             end
% % %             
% % %             if (curPose==30)
% % %                 I =  flip_image(I);
% % %             end
% % %             
% % %             beVerbose = 0;
% % %             ff = [1 1 fliplr(size2(I)-1)];
% % %             RT1_1 = RT1;
% % %             my_inds = [];
% % %             if (frontal)
% % %                 if nn_initialization
% % %                     %                     my_inds = ind_all(comps==1,imm);
% % %                     
% % %                     my_inds = find(all_c>3);
% % %                     my_inds = vl_colsubset(my_inds,RT1,'random');
% % %                     
% % %                     %                 my_inds = my_inds(1:2)
% % %                     RT1_1 = min(RT1,length(my_inds));
% % %                 end
% % %                 %             my_inds =
% % %                 p=shapeGt('initTest',{I},ff,regModel_frontal.model,...
% % %                     regModel_frontal.pStar,regModel_frontal.pGtN,RT1_1,my_inds);
% % %                 testPrm = struct('RT1',RT1_1,'pInit',ff,...
% % %                     'regPrm',regPrm_frontal,'initData',p,'prunePrm',prunePrm,...
% % %                     'verbose',beVerbose);
% % %                 t=clock;[p,pRT] = rcprTest({I},regModel_frontal,testPrm);t=etime(clock,t);
% % %             else
% % %                 if nn_initialization
% % %                     %                     my_inds = ind_all(comps==2,imm)-length(Is_frontal);
% % %                     
% % %                     my_inds = find(all_c<=3)-nFrontal;
% % %                     my_inds = vl_colsubset(my_inds,RT1,'random');
% % %                     %                 my_inds = my_inds(1:2)
% % %                     RT1_1 = min(RT1,length(my_inds));
% % %                 end
% % %                 %             RT1_1 = min(RT1,length(my_inds));
% % %                 p_init=shapeGt('initTest',{I},ff,regModel_profile.model,...
% % %                     regModel_profile.pStar,regModel_profile.pGtN,RT1_1,my_inds);
% % %                 testPrm = struct('RT1',RT1_1,'pInit',ff,...
% % %                     'regPrm',regPrm_profile,'initData',p_init,'prunePrm',prunePrm,...
% % %                     'verbose',beVerbose);
% % %                 t=clock;[p,pRT] = rcprTest({I},regModel_profile,testPrm);t=etime(clock,t);
% % %             end
% % %             figure(1);clf
% % %             vl_tightsubplot(1,2,1);imagesc2(clip_to_bounds(I));
% % %             vl_tightsubplot(1,2,2);imagesc2(clip_to_bounds(I));
% % %             plot(p(1:end/2),p(end/2+1:end),'gd','LineWidth',2);
% % %             ensuredir('~/notes/images/2015_2_10');
% % %             saveas(gcf,['~/notes/images/2015_2_10/' curImgData.imageID(1:end-4) '.png']);
% % %             %              disp(['nn_initialization: ',num2str(nn_initialization)])
% % %             pause
% % %             
% % %         end
% % %     end
% % % end
% % % 
