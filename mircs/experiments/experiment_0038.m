%%% Experiment 0038 %%%%%%%%%%%
%%% 26/5/2014 %%%%%%%%%%%%%%%%%
% use the RCPR training data to learn about occluding regions.

if (~exist('initialized','var'))
    initpath;
    config;        
    conf.get_full_image = true;
    load ~/storage/misc/imageData_new;            
    newImageData = augmentImageData(conf,newImageData);
    m = readDrinkingAnnotationFile('train_data_to_read.csv');
    newImageData = augmentGT(newImageData,m);
    subImages = {newImageData.sub_image};    
    addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));    
    initialized = true;
end
addpath('/home/amirro/code/3rdparty/rcpr_v1');
load faceActionImageNames
validIndices = find(img_sel & isValid);
load('data/COFW_train');
load('data/rcpr.mat','regModel','regPrm','prunePrm');
%%
GLOC_basedir = '/home/amirro/storage/data/gloc/';

gt_dir = fullfile(GLOC_basedir,'parts_lfw_funneled_gt_images');
gtFiles = getAllFiles2(gt_dir);

% get some s40 images and their keypoints
load ~/storage/misc/face_images_1.mat
L_xy=load('~/storage/misc/face_images_1_xy.mat');

% load some of the results for the GLOC dataset
xy_gt = {};
imgs_gt = {};
n = 0; t= 0;
while n < 1000
%for t = 1:100
    t = t+1;
    [pathstr,name,ext] = fileparts(gtFiles{end-t});
    gtPath =  fullfile('~/storage/lfw_keypoints_piotr',[name '.mat']);
    if (~exist(gtPath,'file')),continue;end;        
    n = n+1
    load(gtPath);
    xy_gt{n} = reshape(res(1:58),[],2);
    imgs_gt{n} = imread(gtFiles{end-t});
end
face_masks = cellfun2(@(x) squeeze(x(:,:,2))>0,imgs_gt);
% face_masks = cellfun2(@(x) squeeze(x(:,:,3))==0,imgs_gt);
% 
% fff = 1;
% A = zeros(250/fff,250/fff);
% % 
% for k = 1:length(xy_gt)    
%     A = A+accumarray(round(fliplr(xy_gt{k}/fff)),1,size(A));  
% end
% figure,imagesc2(A)
% % 
% all_xy = {};
% for k = 1:length(xy_gt)
%     plotPolygons(xy_gt{k},'r.');
% %     all_xy{k} = xy_gt{k}(:);
% end
% 
% 
%%


debug_ = false;
obj_masks = {};
seg_masks = {};
%%
for u = 1:length(face_images_1)
u 
I = face_images_1{u};
xy = L_xy.xys{u};

[M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,newImageData(validIndices(u)),1.5,false);
% if (landmarks.s > -.3)
%     continue;
    % end

% find a transformation from cv2tr
n_gt = 50;

D = 150; % final size of output
% resizeFactor = D/size(I,1);
resizeFactor = 1;
xy = xy*resizeFactor;
I = imResample(I,resizeFactor,'bilinear');
err = zeros(1,n_gt);
for t = 1:100;%length(imgs_gt)
%     if (mod(t,100)==0),disp(t),end;
    base_points = xy_gt{t}*resizeFactor;
    input_points = xy;
    T = cp2tform(base_points,input_points,'projective');
    input_points_t = tforminv(T,input_points(:,1),input_points(:,2));
    err(t) = sum(sum((input_points_t-base_points).^2,2));
%     clf; plotPolygons(base_points,'r+');plotPolygons(input_points_t,'g*');
%     pause;
end

[r,ir] = sort(err,'ascend');

% profile on
R = zeros(size2(I));
for it = 1:n_gt
    t = ir(it);
%     if (mod(it,30)==0),disp(it),end;
    base_points = xy_gt{t}*resizeFactor;
    input_points = xy;
    T = cp2tform(base_points,input_points,'projective');
    curMask = imResample(face_masks{t},resizeFactor,'nearest');
    A = imtransform(curMask,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)]);    
    R = R+A;     
end
obj_mask = (R/n_gt);

obj_mask = obj_mask;
obj_mask(obj_mask<.3) = 0;
obj_mask(obj_mask>.9) = 1;
obj_mask = addBorder(obj_mask,3,0);

img_lab = vl_xyz2luv(vl_rgb2xyz(im2single(I)));
img_c = I;
nIterations = 7; nComponents =2;
[seg_mask,energies] = st_segment(im2uint8(img_c),obj_mask,.5,nIterations,nComponents);




% profile viewer
% toc

% seg_mask = obj_mask>.5;
% subplot(2,2,4);displayRegions(I,seg_mask);

obj_masks{u} = {obj_mask};
seg_masks{u} = {seg_mask};

if (debug_)
    clf; 
    subplot(2,2,1),imagesc2(I); plotPolygons(xy,'g+');
    I1 = sc(cat(3,double(obj_mask),I),'prob');
    subplot(2,2,2); imagesc2(I1);
    subplot(2,2,3);
    imagesc2(M);plotPolygons(bsxfun(@minus,face_poly,face_box(1:2)),'g--');
    subplot(2,2,4);imagesc2(I);plotPolygons(fliplr(bwtraceboundary2(seg_mask)),'r--','LineWidth',2');
    pause;
end
end



% GLOC_basedir = '/home/amirro/storage/data/gloc/lfw_funneled';
% allFiles = getAllFiles2(GLOC_basedir);save ~/storage/misc/all_gloc_files allFiles

% 
% nVis = zeros(size(IsTr));
% for k = 1:1:length(IsTr)   
%     phis = phisTr(k,:);
%     [N,D]=size(phis);
%     nfids = D/3;
%     for n=1:N
%         occl=phis(n,(nfids*2)+1:nfids*3);
%         vis=find(occl==0);novis=find(occl==1);
%     end
%     nVis(k) = length(vis);
% end
% 
% [r,ir] = sort(nVis,'ascend');
% %%
% for ik = 1:1:length(IsTr)
%     k = ir(ik);
%     I = IsTr{k};    
%     phis = phisTr(k,:);
%    
%     clf; imagesc2(I); colormap gray;
%     shapeGt('draw',regModel.model,I,phis);
%     curBox = bboxesTr(k,:);
%     curBox(3:4) = curBox(3:4)+curBox(1:2);
%     plotBoxes(curBox,'m--','LineWidth',2);
%     drawnow;pause;    
% end
% 
% %%
% for t = 1:length(allFiles)
%     t
%     curPath = allFiles{t};    
%     [pathstr,name,ext] = fileparts(curPath);
%     if (strcmp(ext,'.txt')), continue, end;
% %     break
%     I = imread(curPath);    
%     [res,bb] = detect_on_set({I},regModel,bboxesTr,regPrm,prunePrm);    
%     shapeGt('draw',regModel.model,I,res{1});
%     plotBoxes(bb,'g--');pause
% end

% save ~/storage/data/faceActionImage_masks obj_masks seg_masks


