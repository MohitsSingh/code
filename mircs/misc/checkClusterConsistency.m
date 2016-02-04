function [ output_args ] = checkClusterConsistency( conf )
%CHECKCLUSTERCONSISTENCY Summary of this function goes here
%   Detailed explanation goes here


[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);
suffix = 'train_full_test';
cluster_iter = 5;
%clusters_path = fullfile(conf.cachedir,['detectors_' num2str(cluster_iter) conf.suffix '.mat']);

cPath = '/home/amirro/code/mircs/data/cache/kmeans_clusters_vistrain_s.mat';
load(cPath);
clusters = clusters([clusters.isvalid]);

discovery_imgs = cat(1,discovery_sets{:});

% check the relative cluster consistency. 
% first, create for each image the list of activations.

all_locs = {};
for k = 1:length(clusters)
    aa = clusters(k).cluster_locs;
    aa(:,13) = k;
    all_locs{k} = aa;
end

all_locs = cat(1,all_locs{:});

QQ = sparse(all_locs(:,11),all_locs(:,13),ones(size(all_locs,1),1));

% QQ in i,j contains number of times cluster j appeared in image i;

figure,imagesc(QQ);
QQ = double(QQ > 0);
% find the maximal correlations between clusters.
cc = QQ'*QQ;
[cc,icc] = sort(cc,'descend');
figure,imagesc(cc(2:end,:));
colorbar;
cor = cc(2,:);
[mx,imx] = sort(cor,'descend');
plot(mx)

% show the clusters that have the maximal correlation.
%%
close all
for f = 1:100
c1 = imx(f);
c2 = icc(2,imx(f));
% % % % % % 
% % % % % % % show the relative locations of the clusters...
% % % % % % locs1 = clusters(c1).cluster_locs;
% % % % % % cx1 = mean(locs1(:,[1 3]),2);
% % % % % % cy1 = mean(locs1(:,[2 4]),2);
% % % % % % xy1 = [cx1,cy1];
% % % % % % locs2 = clusters(c2).cluster_locs;
% % % % % % cx2 = mean(locs2(:,[1 3]),2);
% % % % % % cy2 = mean(locs2(:,[2 4]),2);
% % % % % % xy2 = [cx2,cy2];
% % % % % % 
% % % % % %  commonImages = unique(find(QQ(:,c1) & QQ(:,c2)));
% % % % % %  
% % % % % %  diffs = {};
% % % % % %  pdiff = 0;
% % % % % %  for ic = 1:length(commonImages)
% % % % % %  loc1_i = xy1(locs1(:,11)==commonImages(ic),:);
% % % % % %  loc2_i = xy2(locs2(:,11)==commonImages(ic),:);
% % % % % %  for i0 = 1:size(loc1_i)
% % % % % %      for i1 = 1:size(loc2_i)
% % % % % %          pdiff = pdiff+1;
% % % % % %          diffs{pdiff} = loc2_i(i1,:)-loc1_i(i0,:);
% % % % % %      end
% % % % % %  end
% % % % % %  end
% % % % % %  
% % % % % %  diffs = cat(1,diffs{:});
% % % % % %  
% % % % % %  plot(diffs(:,1),diffs(:,2),'r+')
% % % % % %  
% % % % % %  pause;
% % % % % % 
% % % % % %  
% % % % % % end
% % % % % % 
% % % % % % 
% % % % % % % imshow([clusters(c1).vis;clusters(c2).vis])
% % % % % % 
% % % % % % % check accidental splitting
% % % % % % % % x1 = clusters(c1).cluster_samples;
% % % % % % % % x2 = clusters(c2).cluster_samples;
% % % % % % x_ = [clusters(c1).cluster_samples,clusters(c2).cluster_samples];
% % % % % % loc_ = [clusters(c1).cluster_locs;clusters(c2).cluster_locs];
% % % % % % 
% % % % % % conf_t = conf;
% % % % % % conf_t.clustering.cluster_ratio = size(x_,2)/2;
% % % % % % toSave = 0;
% % % % % % 
% % % % % % new_clusters = kMeansClustering(conf_t,x_,loc_,0,[],0);
% % % % % % new_clusters = visualizeClusters(conf_t,discovery_imgs,new_clusters,64,0);
% % % % % % m = clusters2Images(new_clusters,[0 0 0]);
% % % % % % figure,imshow(m);title('after');
figure,imshow([clusters(c1).vis;clusters(c2).vis]);title('before');

% % % p0 =visualizeLocs2(conf,discovery_imgs,clusters(c1).cluster_locs,64,1,0,1);

pause;
end
% visualizeLocs(conf,discovery_imgs,M,64,1,0);
%conf,ids,locs,height,inflateFactor,add_border,draw_rect)
% % % pause;
close all
%%
close all;
for q = 1:20
   q = 1
   
   [clusters1,allImgs,isclass] = visualizeClusters(conf,discovery_imgs,clusters(q),64,0);
   figure,imshow(clusters1.vis)
p0 =visualizeLocs2(conf,discovery_imgs,clusters(q).cluster_locs,64,1,0,0);
imshow(clusters(q).vis)
figure,imshow(multiimage(p0));
pause
end
;
p1 = visualizeLocs2(conf,discovery_imgs,clusters(c2).cluster_locs,64,1,0,1);
%%
figure,imshow(multiImage(p1))
all_locs(all_locs(:,13)==c2,11)


% figure,imagesc(QQ)
% 
% 
% 
% M = all_locs(all_locs(:,11)==38,:);
% figure,imagesc(M)
% 
% [p,p_mask] = visualizeLocs(conf,discovery_imgs,M,64,1,0);


% clusters_path_lite = strrep(clusters_path,'.mat','_lite.mat');
end

