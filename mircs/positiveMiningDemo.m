
%% positive sample mining
%
initpath;
config;
% precompute the cluster responses for the entire training set.
%
conf.suffix = 'positive_mining';

conf.VOCopts = VOCopts;

conf.class_subset = APPLAUDING;

% dataset of images with ground-truth annotations
% start with a one-class case.

[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels] = getImageSet(conf,'test');
conf.detetion.params.detect_max_windows_per_exemplar = 1;

conf.detection.params.max_models_before_block_method = 0;
conf.max_image_size = 100;
conf.clustering.num_hard_mining_iters = 5;
conf.max_image_size = 256;

%%
ids = train_ids(train_labels);
im = getImage(conf,ids{38});
rects = selectSamples(conf,{im});
%save faceRect_0 rects
load faceRect_0;
faceClust = rects2clusters(conf,rects,{im},[],0);
faceTrained = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','face_0','override',false);
figure,imshow(showHOG(conf,faceTrained));
imwrite(showHOG(conf,faceTrained),'posMining/hogWeights.jpg');
mkdir('posMining');

% run first time on positive set.
conf.detection.params.detect_levels_per_octave = 8;
[qq0,q0] = applyToSet(conf,faceTrained,ids,[],'face_0_run0','override',false);

a = visualizeLocs2_new(conf,ids,qq0.cluster_locs,'add_border',false);
m_a = multiImage(a(1:25));
figure,imshow(m_a)
imwrite(m_a,'posMining/posDetections.jpg');


% select a random subset of negative images.

figure,imshow(getImage(conf,neg_images{1}))

neg_images = vl_colsubset(col(getNonPersonIds(VOCopts))',500);
% save neg_image neg_images
% load neg_images;
[qq0_neg] = applyToSet(conf,faceTrained,neg_images,[],'face_0_run0_neg','override',false);
a_neg = visualizeLocs2_new(conf,neg_images,qq0_neg.cluster_locs,'add_border',false);
[mm,~,x,y] = multiImage(a_neg(1:25));
figure,imshow(mm);
imwrite(mm,'posMining/negDetections.jpg');

% select an informative sub-patch.
conf2 = conf;
conf2.detection.params.init_params.sbin = 5;
conf2.features.winsize = [4 4];
subRect1 = selectSamples(conf2,a(1));

conf2.detection.params.detect_min_scale = .5;
[X,locs] = samplePatches(conf2,a(1),.1);

locs_ = locs;
locs_(:,11) = 1;
subPatches = visualizeLocs2_new(conf2, a(1), locs_);
subPatches = multiImage(subPatches);
imwrite(subPatches,'posMining/subPatches.jpg');

figure,imshow(a{1}); hold on; plotBoxes2(locs(:,[2 1 4 3]))
c= makeClusters(X,locs);
subPatchClust = rects2clusters(conf2,subRect1,a(1),[],1);

% [X,locs] = samplePatches(conf,ids,ovp)
save subRect_0 subRect1
conf2.detection.params.detect_add_flip = false;
% train it on the negative set again...
conf2.detection.params.detect_min_scale = .5;
subPatchTrained = train_patch_classifier(conf2,c,getNonPersonIds(VOCopts),'suffix','subPatchs_0','override',true);

[qq1_pos] = applyToSet(conf2,subPatchTrained,a,[],'sub_0_pos','override',true,'uniqueImages',true);
[qq1_neg] = applyToSet(conf2,subPatchTrained,a_neg,[],'sub_0_neg','override',true,'uniqueImages',true);
%%
cluster_sel = 22;
toShow = 1;

if (toShow)
    eye_pos = visualizeLocs2_new(conf,a,qq1_pos(cluster_sel).cluster_locs,'draw_rect',false);
    eye_neg = visualizeLocs2_new(conf,a_neg,qq1_neg(cluster_sel).cluster_locs,'draw_rect',false);
        
    pos_inds = qq1_pos(cluster_sel).cluster_locs(:,11);
    neg_inds = qq1_neg(cluster_sel).cluster_locs(:,11);
    n = 25;
    [mm_true,~,x_true,y_true] = multiImage(a(pos_inds(1:n)),false);
    [mm_false,~,x_false,y_false] = multiImage(a_neg(neg_inds((1:n))));
    pos_boxes = shiftBoxes(qq1_pos(cluster_sel).cluster_locs(1:n,:),x_true(1:n),y_true(1:n));
    neg_boxes = shiftBoxes(qq1_neg(cluster_sel).cluster_locs(1:n,:),x_false(1:n),y_false(1:n));
    
    figure,vl_tightsubplot(1,1,1); imshow(mm_true);
            
    hold on;
    plotBoxes2(pos_boxes(:,[2 1 4 3]),'color','g','LineWidth',2);
    
    figure,vl_tightsubplot(1,1,1);imshow(mm_false);
    hold on;
    plotBoxes2(neg_boxes(:,[2 1 4 3]),'color','g','LineWidth',2);
    
end
[Z_pos,pts_pos] = createConsistencyMaps(qq1_pos,[64 64],[],inf,[15 3]);
[Z_neg,pts_neg] = createConsistencyMaps(qq1_neg,[64 64],[],inf,[15 3]);
if toShow
    figure,imshow(multiImage(jettify(Z_pos)));title('pos!');
    figure,imshow(multiImage(jettify(Z_neg)));title('neg!');
end
%
imwrite(multiImage(jettify(Z_pos)),'posMining/pos_pdf.jpg');
imwrite(multiImage(jettify(Z_neg)),'posMining/neg_pdf.jpg');
%
%

pts_pos = pts_pos{cluster_sel};
pts_neg = pts_neg{cluster_sel};

pts = [pts_pos ;pts_neg]';

labels = -ones(1,size(pts,2));


labels(size(pts_pos,1)+1:end) = 1;


[mu,sigmas,pi_,LL] = emAlgorithm(pts,labels,2,Z_pos{cluster_sel}+Z_neg{cluster_sel});
figure,plot(LL,'.-')

figure(1);
imagesc(Z_pos{cluster_sel});colormap hot; hold on;
plot(pts(1,:),pts(2,:),'r.');
plot(mu(1,1),mu(2,1),'md','MarkerSize',3,'LineWidth',5); % negative
plot(mu(1,2),mu(2,2),'gd','MarkerSize',3,'LineWidth',5); % positive 
legend({'data','neg','pos'});

% now, sort the points according to likelyhood for the first component...
% all that matters is the covariance matrix and distance from the centroid.
%%
sigs_inv = sigmas;
sigs_det = zeros(1,2);
for j = 1:2
    sigs_inv(:,:,j) = inv(sigmas(:,:,j));
    sigs_det(j) = det(sigmas(:,:,j));
end
% calculate the posterior for each point...
a_ = (2*pi)^2/2;
P_i_j = zeros(size(pts_pos,1),2); %non-normalized
for iX = 1:size(pts_pos,1)
    for iC = 1:2
        x_d = pts_pos(iX,:)'-mu(:,iC);
        P_i_j(iX,iC) = pi_(iC)*exp(-.5*x_d'*sigs_inv(:,:,iC)*x_d)/(sigs_det(iC)*a_);
    end
end

gamma_ij = bsxfun(@rdivide,P_i_j,sum(P_i_j,2));
newScores = gamma_ij(:,2);

%%

[r,ir] = sort(newScores,'descend');

figure,imshow(multiImage(eye_pos(1:25)))

figure,imshow(multiImage(eye_pos(ir(1:25)))); title('sorted');

addpath('/home/amirro/code/3rdparty/bmp_plot/');
%% visualize the newly sorted detections to show location is actually better.
% % qq1_pos_modified = qq1_pos(cluster_sel);
% % qq1_pos_modified.cluster_locs = qq1_pos_modified.cluster_locs(ir,:);
% % visualizeLocs2(conf,a,qq1_pos_modified.cluster_locs,'draw_rect',true);
% % visualizeLocs2(conf,a,qq1_pos(cluster_sel).cluster_locs,'draw_rect',true);
n = 50;
[mm_true,~,x_true,y_true] = multiImage(a(pos_inds(1:n)),false);
pos_boxes = shiftBoxes(qq1_pos(cluster_sel).cluster_locs(1:n,:),x_true(1:n),y_true(1:n));

figure, imshow(mm_true);

hold on;
plotBoxes2(pos_boxes(:,[2 1 4 3]),'color','g','LineWidth',2); title('orig');

[mm_true,~,x_true,y_true] = multiImage(a(pos_inds(ir(1:n))),false);
pos_boxes = shiftBoxes(qq1_pos(cluster_sel).cluster_locs(ir(1:n),:),x_true(1:n),y_true(1:n));

figure,imshow(mm_true);

hold on;
plotBoxes2(pos_boxes(:,[2 1 4 3]),'color','g','LineWidth',2); title('sorted');

% title(num2str(curLikelyhood));
