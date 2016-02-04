function svm_model = secondaryClustering_mdf(conf,dets,train_set,gt_labels,cls,r_true_all)
% secondaryClustering(conf,top_dets(2),test_set,gt_labels);

suffix = ['_discxy_128_1' cls];

% TODO - limit the number of positive examples to a few (5 or 10)
% to keep the cluster purity high. This way, the secondary clustering will
% not become confused with irrelevant information.

visPath = fullfile(conf.cachedir,['phase1_vis' suffix '.mat']);
img_side = 64;
k_topdets = conf.clustering.secondary.sample_size;
inflate_factor = conf.clustering.secondary.inflate_factor;
add_border = 0;
if (~exist(visPath,'file'))
    f = find(gt_labels);
    cur_locs = cat(1,dets.cluster_locs);
    cur_locs = cur_locs(1:min(k_topdets,size(cur_locs,1)),:);
            
    det_inds = cur_locs(:,11);
    [tf] = ismember(det_inds,f);
    cc = cur_locs(tf,:);
    cc = cc(1:min(2*15,size(cc,1)),:);
    r_true = visualizeLocs2(conf,train_set,cc ,img_side,inflate_factor,add_border,0);
%     figure,imshow(multiImage(r_true));
    cc = cur_locs(~tf,:);
    cc = cc(1:end,:);
    r_false = visualizeLocs2(conf,train_set,cc,img_side,inflate_factor,add_border,0);
    save(visPath,'r_true','r_false');
else
    load(visPath);
end 
%%

samples_true = {};
samples_false = {};
conf.detection.params.init_params.sbin = 8;
conf.features.winsize = 8;

for k = 1:length(r_true)
    X = allFeatures( conf,r_true{k});
    samples_true{k} = X(:,1);
end
for k = 1:length(r_false)
    k
    X = allFeatures( conf,r_false{k});
    samples_false{k} = X(:,1);
end

samples_true = cat(2,samples_true{:});
samples_false = cat(2,samples_false{:});

all_samples = [samples_true,samples_false];
all_samples = vl_homkermap(all_samples, 1, 'kchi2', 'gamma', .5) ;
all_labels = ones(size(all_samples,2),1);
all_labels(size(samples_true,2)+1:end) = -1;
C = .1;
w1= 1;
svm_model = svmtrain(all_labels, all_samples',sprintf(['-s 0 -t 0 -c' ...
    ' %f -w1 %.9f -q'], C, w1));


