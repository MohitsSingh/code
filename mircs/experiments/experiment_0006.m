%%%%%% Experiment 6 %%%%%%%
% Oct. 28, 2013

% Attempt to make a class-specific saliency measure. For example, do it for
% faces. 

% Start by extracting detected faces on the stanford 40 dataset.

initpath;
config;

% cluster data
L_clusters = load('/home/amirro/mircs/experiments/common/faceClusters_big.mat');
load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat');
clusters = L_clusters.clusters;
X_big = fevalArrays(cat(4,ims{:}),@(x)col(fhog(im2single(x))));

[~,ims_,~] = makeClusterImages(ims,L_clusters.C',L_clusters.IDX',X_big,[],10);
% cluster images...


%X = fevalArrays(ims1,@(x)col(hog(im2single(x))));
resultDir = ['~/mircs/experiments/' mfilename];
ensuredir(resultDir);
conf.class_subset = conf.class_enum.DRINKING;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
conf.get_full_image =false;


%% already ran on all stanford 40, show some results
iModel =1; % 
subs_ = {};
conf.get_full_image = false;
T =train_ids';
T = train_ids(train_labels);
%T = T(1:1090);
% T = vl_colsubset(T,1000,'Uniform');
rects_to_crop = -ones(length(T),11);
scores = -1000*ones(size(T));
for k = 1:length(T)
    k
    resFile = fullfile('~/storage/faces_s40_big_x2',strrep(T{k},'.jpg','.mat'));
    load(resFile); %-->res
        curImage = getImage(conf,T{k});
%         colors = {'r','g','b'}';
%         clf; imagesc(curImage);axis image; hold on;
%         for iModel = 1:3
    dss = res(iModel).ds;
    if (isempty(dss))
        continue;
    end
    [s,is] = max(dss(:,end));
    scores(k) = s;
    dss = dss(is,:);
    rects_to_crop(k,1:4) = dss(1:4);
    rects_to_crop(k,11) = k;
    rects_to_crop(k,12) = dss(end);
%             plotBoxes2(dss([2 1 4 3]),[colors{iModel} '--'],'LineWidth',2);
%         end
%         pause
end

isBad = rects_to_crop(:,1) == -1;
rects_to_crop = rects_to_crop(~isBad,:);


%% 
load /home/amirro/mircs/experiments/common/faces_cropped_new.mat;
load ~/code/mircs/imageData_new.mat
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
imgs = faces.test_faces(test_labels);
%imgs = multiCrop(conf,imgs,[],[80 80]);
imgs = cellfun2(@(x) imresize(x,[80 80],'bilinear'),imgs);
%imgs = multiCrop(conf,T,rects_to_crop,[80 80]);
% showSorted(imgs,rects_to_crop(:,12),300);

% find the "best" cluster for each image. 
X = fevalArrays(cat(4,imgs{:}),@(x)col(fhog(im2single(x))));

% % vis_x = fevalArrays(cat(4,imgs{1:50}),@(x)hogDraw(fhog(im2single(x).^2),15,1));
% % 
% % vis_w = zeros(150,150,10);
% % for k = 1:length(clusters)
% %     vis_w(:,:,k) = hogDraw(reshape(clusters(k).w.^2,[10 10 32]),15, 1);
% % end
    

%%
%vis_clusters = fevalArrays(cat(2,clusters.w),@(x)hogDraw(fhog(im2single(x).^2),15,1));
for k = 1:length(imgs)        
   curX = X(:,k);
   dists = zeros(size(clusters));
   for iCluster = 1:length(clusters)
       d = l2(curX',clusters(iCluster).cluster_samples');
       dists(iCluster) = min(d);       
   end
   
   clf; 
   subplot(2,2,1); imagesc(imgs{k}); axis image;
   
   [r,id] = min(dists);
   
   subplot(2,2,2); 
   V = vis_w(:,:,id);
   %hogDraw(reshape(clusters(id).w.^2,[10 10 32]),15, 1);
   imagesc(V); axis image;title(sprintf('cluster: %d, purity: %3.3f',id,1-r./mean(dists)));   
   subplot(2,2,3); imshow(ims_{id});
   pause
end


[s,is] = sort(rects_to_crop(:,12),'descend');
%%

addpath(genpath('/home/amirro/code/3rdparty/ihog-master/ihog-master'));

%vis_clusters = fevalArrays(cat(2,clusters.w),@(x)hogDraw(fhog(im2single(x).^2),15,1));
for q = 1:length(imgs)
    
   nnn = 100;
    
   k = is(q);
   if isempty(strfind(T{k},'drink'))
       continue;
   end
   curX = X(:,k);
   %dists = zeros(size(clusters));
   dists = l2(curX',X_big');  
   [r,id] = sort(dists,'ascend');
   clf;
   subplot(2,2,1);
   imagesc(imgs{k}); axis image;       
   subplot(2,2,2); 
   montage2(cat(4,ims{id(1:nnn)}),struct('hasChn',1));
   score = sum(exp(-r(1:nnn)/100));title(num2str(score));
   
   diffs = bsxfun(@minus,X_big(:,id(1:nnn)),curX);
   %diffs = sum(diffs.^2,2);
   diffs = mean(abs(diffs),2);
   m = reshape(diffs,10,10,32);
%    ihog = invertHOG(m);
%    V = hogDraw(m,15,1);
%    subplot(2,2,3); imagesc(V); colormap jet; axis image;
%    subplot(2,2,4);
%    imagesc(ihog); axis image;
    
   % extract features:
   % edges, colordistributions, lpp features, etc. 
   % even patches!!
   imSet = cat(4,ims{id(1:nnn)});
  
   edges = fevalArrays(imSet,@(x) edge(rgb2gray(x),'canny'));
    dists = fevalArrays(edges,@bwdist);
   E = mean(edges,3);
   subplot(2,2,3); imagesc(mean(exp(-dists/10),3)); axis image;
   
   %montage2(edges)
   
    

   %imagesc(sum(m,3)); colormap jet; axis image;
   disp(score)
   
   %imagesc(ims{id}); axis image;
   pause
end
%%
%%

[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

image_ids = train_ids(train_labels);
% image_ids = train_ids(1:67);
collectResults(conf,image_ids);




