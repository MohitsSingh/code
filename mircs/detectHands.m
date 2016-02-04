handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
addpath(handDetectionBaseDir );
addpath(fullfile(handDetectionBaseDir,'code/globalPb'));

d
a = train_faces(t_train);
figure,imshow(a{1})

train_face_labels = {};
for q = 1:length(a)
    [gPb_orient, gPb_thin, textons] = globalPb(im2double(a{q}), 'tmp.mat', 1.0);
%     figure,imagesc(gPb_thin);
%     figure,imagesc(textons)
    ucm = contours2ucm(gPb_orient);
     k = 0.1; %64; %100
    bdry = (ucm >= k);
    train_face_labels{q} = bwlabel(ucm <= k);
%     figure,imagesc(labels)
%     figure,imagesc(a{1})
end
    
train_ids_d = train_ids(train_dets.cluster_locs(:,11));
true_train_images = train_ids_d(train_labels);
%
faceBoxes_train = train_dets.cluster_locs;
faceBoxes_train_t = faceBoxes_train(train_labels,:);

myProcess_gPb(conf,train_ids_d,handDetectionBaseDir);

% for k = 11:50
%     clf;
%     imshow(getImage(conf,true_train_images{k}));
%     hold on;
%     plotBoxes2(faceBoxes_train_t(k,[2 1 4 3]));
%     pause;
% end