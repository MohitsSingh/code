%% Experiment 0062 - 23/3/2015
%-----------------------------

% 1. Create a facial landmark localization with "good enough" performance
% 2. Refine the location of facial landmarks locally, around the mouth
% 3. create a model to best predict the configuration of the action object

% rmpath('/home/amirro/code/my_landmark_localization');

addpath('~/code/3rdparty/vedaldi_detection/');

isinitialized = true
if (~isinitialized)
    load ~/storage/data/face_data.mat
end

dets = cat(1,face_data.face_det);
%%
% 1:left, 2:right
% 3: half-right, 4:half-left
% 5,6 - near-frontal

% start with near frontal as an example.
min_face_score = 3;
img_sel = true(size(face_data));
required_poses = [5 6];

all_dets = cat(1,face_data.face_det);

img_sel = ismember(all_dets(:,5),required_poses) & all_dets(:,6)>=min_face_score;
for t = 1:1:length(face_data)       
    if (~img_sel(t)),continue,end
    pts = face_data(t).kp_data.pts;
    yaw = face_data(t).pose.yaw;
    bbox = face_data(t).face_det;
    det_pose = bbox(5);
    det_score = bbox(6);
        
    I = imread(face_data(t).image_path);    
    sz = size2(I);
    if yaw < 0  && 0
        bbox = flip_box(bbox,sz);
        pts = flip_pt(pts,sz);
        I = flip_image(I);
    end
    clf; figure(1); imagesc2(I);
    plotBoxes(bbox);
    plotPolygons(pts,'g.');
    drawnow;
    pause
end
face_data_t = face_data(img_sel);
train_kp_detector(face_data_t)


% 0: create a data-structure containing for each image its 
% file-name, face bounding box, keypoints and other metadata

% 
% 
% [paths,names] = getAllFiles('~/storage/data/aflw_cropped_context','.jpg');
% L_pts = load('~/storage/data/ptsData');
% ptsData = L_pts.ptsData(1:end);
% poses = L_pts.poses(1:end);
% requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
% %
% rolls = [poses.roll];
% pitches = [poses.pitch];
% yaws = [poses.yaw];
% goods = abs(rolls) < 30*pi/180;
% [u,iu] = sort(rolls,'descend');
% edges = [0 20 45 90];
% [b,ib] = histc(180*abs(yaws)/pi,edges);
% poseMap = [90 -90 30 -30 0 0];
% % load the dpm detections on aflw.
% dpmDetsPath = '~/storage/data/aflw_cropped_context/dpm_detections.mat';
% if (exist(dpmDetsPath,'file'))
%     load(dpmDetsPath);
% else 
%     ress = zeros(length(paths),6);
%     id = ticStatus( 'loading paths', .5);
%     for p = 1:length(paths)
%         detPath = j2m('~/storage/aflw_faces_baw',paths{p});
%         load(detPath);
%         nBoxes = size(res.detections.boxes,1);
%         if (nBoxes > 0)
%             ress(p,:) = res.detections.boxes(1,:);
%         end
%         tocStatus(id,p/length(paths));
%     end
%     save(dpmDetsPath,'ress');
% end
% 
% 
% face_data = struct('image_path',{},'pose',{},'kp_data',{},'face_det',{});
% 
% for t = 1:length(paths)
%     %curImg = imread(paths{t});
%     face_data(t).image_path = paths{t};
%     face_data(t).kp_data = ptsData(t);
%     face_data(t).pose = poses(t);
%     face_data(t).face_det = ress(t,:);
% end

% save ~/storage/data/face_data.mat face_data

