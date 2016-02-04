% compile.m should work for Linux and Mac.
% To Windows users:
% If you are using a Windows machine, please use the basic convolution (fconv.cc).
% This can be done by commenting out line 13 and uncommenting line 15 in
% compile.m
% compile;

% load and visualize model
% Pre-trained model with 146 parts. Works best for faces larger than 80*80
load face_p146_small.mat

% % Pre-trained model with 99 parts. Works best for faces larger than 150*150
%load face_p99.mat
% load /home/amirro/code/3rdparty/face-release1.0-basic/models_CVPR2012/multipie_independent.mat

% % Pre-trained model with 1050 parts. Give best performance on localization, but very slow
% load multipie_independent.mat

% disp('Model visualization');
% visualizemodel(model,1:13);
% disp('press any key to continue'); 
% pause;
% 

% 5 levels for each octave
model.interval = 5;
% set up the threshold
model.thresh = min(-.9, model.thresh);

% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13 
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end

% ims = dir('images/*.jpg');
% imgDir ='/home/amirro/data/Stanford40/JPEGImages/';
% ims = dir(fullfile(imgDir,'drinking*.jpg'));
% ims = mSide;
ims = m1_train_true;
for i = 1:length(ims),
    fprintf('testing: %d/%d\n', i, length(ims));
%     im = imread(fullfile('images',ims(i).name));
    im = ims{i};
    im = (getImage(conf,train_ids{qq1(k).cluster_locs(f_true(i),11)}));
%     im = imresize(im,2);
%     clf; imagesc(im); axis image; axis off; drawnow;
    
    tic;
    bs = detect(im, model, model.thresh);
    bs = clipboxes(im, bs);
    bs = nms_face(bs,0.3);
    
    if (isempty(bs))
        disp('no faces detected...');
        pause;
        continue;
    end
        
    dettime = toc;
    
    % show highest scoring one
    figure,showboxes(im, bs(1),posemap),title('Highest scoring detection');
    % show all
    figure,showboxes(im, bs,posemap),title('All detections above the threshold');
    
    fprintf('Detection took %.1f seconds\n',dettime);
    disp('press any key to continue');
    pause;
    close all;
end
disp('done!');
