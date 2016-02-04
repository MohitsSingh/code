initpath;
config;
% conf.max_image_size = 200;
conf.suffix = 'train_mdf_tt_new';

% dataset of images with ground-truth annotations
% start with a one-class case.
[train_ids,train_labels] = getImageSet(conf,'train',2,0);

% [clusters,estQuality] = findGoodPatches(conf,train_ids(1:2:end),train_labels(1:2:end));

% for testing, start from second image and get every second image
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);
cls = 15; % class of person;
[ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_train']),'%s %d');
ids = ids(t==-1); % persons non gratis
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);
natural_set = {ids(1:2:end),ids(2:2:end)};

% create a k-means dictionary for the sampling process
% dict = learnDictionary(conf,train_ids);
% conf.dict = dict;

clustering3(conf,discovery_sets,natural_set);

L = load('/home/amirro/storage/data/cache/detectors_5train_mdf_tt_new');
cur_suffx=  [conf.suffix '_test'];
[test_ids,test_labels] = getImageSet(conf,'test',1,0);

test_dets = getDetections(conf,test_ids,L.clusters,[],cur_suffx,true)

I = getImage(conf,natural_set{1}{1});
figure,imshow(I)
ovp = .5;
conf.detection.params.detect_min_scale = 1;
[X,uus,vvs,scales,t ] = allFeatures( conf,I,ovp);
boxes =uv2boxes(conf,uus,vvs,scales,t);
figure,imshow(I);
hold on;
plotBoxes2(boxes(:,[2 1 4 3]))

%conf,uu,vv,scales,t

[val_ids,val_labels] = getImageSet(conf,'train',2,1);

iter_num = 5;
clusters = [];
[top_dets_val,aps_val] = test_clusters(conf,clusters,iter_num,conf.suffix,val_ids,val_labels,'train_val_test',50);

% now learn a class distribution for each "detector", and sum the
% distributions.

[prec,rec,aps_val,T,M] = calc_aps(top_dets_val,val_labels,[],inf);
[~,ibest_val] =sort(aps_val,'descend');

% try a random forest!
ntrees = 5;
B = TreeBagger(ntrees,M(:,ibest_val(1:20)),val_labels,'NVarToSample','all');
plot(oobError(B));
y = B.predict(M)
y = cellfun (@(x) (str2num(x)), y);
% find the maximal information gain for each detector.
% plot((rec))
nDetectors = length(aps);
stumps = zeros(1,nDetectors);
p_lefts= zeros(1,nDetectors);
p_rights = zeros(1,nDetectors);

for k = 1:nDetectors
    cur_detector = k;
    %t = T(:,cur_detector);
    grades = top_dets_val(k).cluster_locs(:,12);
    t = val_labels(top_dets_val(k).cluster_locs(:,11));
    [gain_curve,p_left,p_right] = information_gain(t);
    [m,im] = max(gain_curve);
    stumps(k) = grades(im);
    p_lefts(k) = p_left(im);
    p_rights(k) = p_right(im);
end

P = [p_lefts;p_rights]';

k_choice = 1
suffix2 = '_mdf_t1';
% suffix2 = '_mdf_t1_comb';

% get all the positive samples.
[trues,falses] = get_top_samples(conf,top_dets_val(ibest_val(k_choice)),val_ids,val_labels,suffix2,1,10);
suffix2 = '_mdf_true'
[trues,inds] = get_top_samples2(conf,top_dets_val(ibest_val(k_choice)),val_ids,val_labels,suffix2,0,10);

imshow(multiImage(falses{1}))

%find common features.
[clusters_] = find_common_features(conf,trues(10),inds,0);

% now, apply on part of the test set
[test0_ids,test0_labels] = getImageSet(conf,'test',2,0);

iter_num = 5;
clusters = [];
[top_dets_test0,aps_test0] = test_clusters(conf,[],iter_num,conf.suffix,test0_ids,test0_labels,'train_test_0');
[prec_0,rec_0,aps_test0,T_0,M_0] = calc_aps(top_dets_test0,test0_labels,[],inf);

y = B.predict(M_0(:,ibest_val(1:20)));
y = cellfun (@(x) (str2num(x)), y);
plot(test0_labels)
figure,plot(y)

sum(y.*test0_labels)
%%
Z = zeros(length(test0_labels),2);
nZ = zeros(size(Z,1),1);
bests = ibest_val(1:20);
for k = 1:length(top_dets_test0)
    if (~any(find(bests==k)))
        %         Z(k,1) = -2;
        continue;
    end
    
    % find all voters for this label, and sum the probabilities according
    % to the threshold.
    k
    for iz = 1:size(Z,1)
        is_z = find(top_dets_test0(k).cluster_locs(:,11)==iz);
        nZ(iz) = nZ(iz)+length(is_z);
        ttt = 2-(top_dets_test0(k).cluster_locs(is_z,12)>=stumps(k));
        for kk = 1:min(1,length(ttt))
            Z(iz,ttt(kk)) = Z(iz,ttt(kk))+P(k,ttt(kk));
        end
        %         if ttt(1)==1
        %             Z(iz,ttt)
        %         end
        %         for kk = 1:length(ttt)
        %             Z(iz,ttt(kk)) = Z(iz,ttt(kk))+P(k,ttt(kk));
        %         end
    end
end

Q = bsxfun(@rdivide,Z,sum(Z,2));
Q(isnan(Q)) = 0;
[q,iq] = sort(Q(:,1),'descend');

plot(test0_labels(iq));

det_s = top_dets_test0(1);
det_s.cluster_locs = zeros(length(q),12);
% iq(q==0) = randperm(iq(q==0));
det_s.cluster_locs(:,11) = iq;
det_s.cluster_locs(:,12) = q;

[prec_s,rec_s,aps_s,T_]  = calc_aps(det_s,test0_labels)
plot(rec_s,prec_s);title(num2str(aps_s));
%%
[~,ibest_test0] =sort(aps_test0,'descend');

% detect common features on validation set!
% visualizeLocs2(conf,ids,locs,height,inflateFactor,add_border,draw_rect)
% r = visualizeLocs2(conf,train_set,cur_locs,img_side,inflate_factor,add_border,0);

%%
locWeight =0.1;
svms = learn_fine_grained(conf,trues,falses,locWeight);
k = 1
suffix = ['_discxy_128_1' num2str(k) suffix2];
conf_new = conf;
conf.detection.params.detect_min_scale  = 1;
re_sorted = struct;
k =1;
% for k =k_choice
close all;
aps_ = {};
%weights = 0:.1:1;
% weights = .1%;.1;
% for iw = 1:length(weights)
iw=1;
toshow =1;
k = 1
suffix = ['smirc_test0' num2str(k)];
svms_ = svms%(65:75);
% svms_ = svms([61    52    31    60    23    22    59    25    43    24]);
%svms_ = svms([61    52    31    60    23]);%   22    59    25    43    24]);
% svms_ = svms(im(1:20))
[re_sorted_0,dets_,im] = sort_secondary_fine(conf_new,top_dets_test0(ibest_val(k)),svms_,test0_ids,test0_labels,toshow,suffix,locWeight,clusters_);
% %     betas = perform_calibration(re_sorted_0,test0_labels);
% and check the results
[prec2,rec2,aps2] = calc_aps(dets_,test0_labels);
[prec1,rec1,aps1] = calc_aps(top_dets_test0((ibest_val(k))),test0_labels);
aps_{iw} = aps2;
% endqbit
%
% aps_1 = cat(1,aps_{:});
% plot(weights,aps_1);

figure,plot(rec2,prec2); hold on;plot(rec1,prec1,'r');
legend({['phase 2; ap=' num2str(aps2,'%1.3f')],['phase 1; ap=' num2str(aps1,'%1.3f')]});
%%
re_sorted(k).dets_ = dets_;
%     pause;
% end

%%
% now, and test on remainder of test-set
[test1_ids,test1_labels] = getImageSet(conf,'test',2,1);

iter_num = 5;
clusters = [];
% test just this one...
[top_dets_test1,aps_test1] = test_clusters(conf,[],iter_num,conf.suffix,test1_ids,test1_labels,'train_test_1');

[~,ibest_test1] =sort(aps_test1,'descend');

suffix = ['_discxy_128_1' num2str(k) suffix2];;
conf_new = get_secondary_clustering_conf(conf);
smirc_clusters{k} = train_patch_classifier(conf_new,[],[],1,suffix);
re_sorted = struct;
for k =k_choice
    close all;
    toshow =1;
    suffix = ['smirc_test1' num2str(k)];
    %     im = 1:length(smirc_clusters{k});
    [re_sorted_,dets_,im] = sort_secondary(conf_new,top_dets_test1(ibest_val(k)),smirc_clusters{k},test1_ids,test1_labels,toshow,suffix,[]);
    % and check the results
    [prec2,rec2,aps2] = calc_aps(dets_,test0_labels);
    [prec1,rec1,aps1] = calc_aps(top_dets_test0((ibest_val(k))),test0_labels);
    figure,plot(rec2,prec2); hold on;plot(rec1,prec1,'r');
    legend({['phase 2; ap=' num2str(aps2,'%1.3f')],['phase 1; ap=' num2str(aps1,'%1.3f')]});
    re_sorted(k).dets_ = dets_;
    %     pause;
end

%% scratch
%
%
% betas = zeros(3,2);
% for k = 1:3%length(top_dets)
%     k
%     tt = top_dets(ibest_val(k)).cluster_locs;
%     scores = tt(:,12);
%     os = val_labels(tt(:,11));
%     os(6:end) = 0;
%     beta = esvm_learn_sigmoid(scores, os);
% %     xs = linspace(min(scores),max(scores),1000);
% %     fx = @(x)(1./(1+exp(-beta(1)*(x-beta(2)))));
%     betas(k,:) = beta;
% end


%
%  plot(xs,fx(xs),'b','LineWidth',2)
%     hold on
%     plot(scores,os,'r.','MarkerSize',14)
% beta = esvm_learn_sigmoid(scores, os)

%
% top_dets2 = top_dets(ibest_val(1:20));
% for k = 1:3
%     k
%     scores = top_dets2(k).cluster_locs(:,12);
%     fx = @(x)(1./(1+exp(-betas(k,1)*(x-betas(k,2)))));
%     scores = fx(scores);
%     top_dets2(k).cluster_locs(:,12) = scores;
% end
%
% top_dets3 = top_dets2(1);
% q = [];
% for k = 1:3
%     q = [q;top_dets2(k).cluster_locs];
% end
% top_dets3.cluster_locs = q;
%
% [prec,rec,aps] = calc_aps(top_dets3,test0_labels,[],inf);
