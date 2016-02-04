
function [re_topdets,dets,im] = sort_secondary_fine(conf,dets,svms,train_set,...
    gt_labels,toshow,suffix,betas,iim);
img_side = 64;
k_topdets = conf.clustering.secondary.sample_size;
inflate_factor = conf.clustering.secondary.inflate_factor;
add_border = 0;

use_best = 1;

% f = find(gt_labels);
% cluster_sel = 1;

cur_locs = cat(1,dets.cluster_locs);
k_topdets = min(k_topdets,size(cur_locs,1));
cur_locs = cur_locs(1:k_topdets,:);
visPath_det =  fullfile(conf.cachedir,['phase2_vis4det_hkm' suffix '_' num2str(k_topdets) '.mat']);
if (~exist(visPath_det,'file'))
    r = visualizeLocs2(conf,train_set,cur_locs,img_side,inflate_factor,add_border,0);
    save(visPath_det,'r');
else
    load(visPath_det);
end

tic
conf.detection.params.detect_max_windows_per_exemplar = 1;
conf.detection.params.max_models_before_block_method = 5;
% conf.detection.params.detect_min_scale =1;
% conf.detection.params.detect_exemplar_nms_os_threshold = .5;
% conf.detection.params.detect_min_scale = .5;
w = svm_model.SVs'*svm_model.sv_coef;

ws = w(:);
b = svm_model.rho;
sv = svm_model.SVs';

re_dets = dets;
conf_new = get_secondary_clustering_conf(conf);
for k = 1:length(r)
    X = allFeatures(conf_new,r{k});
    X = X(:,1);
    X = vl_homkermap(X, 1, 'kchi2', 'gamma', .5) ;
    re_dets.cluster_locs(k,12) = X'*ws-b;
    re_dets.cluster_locs(k,1:4) = [1 1 64 64];
       re_dets.cluster_locs(k,7) = 0;
end



re_dets.cluster_locs(length(r)+1:end,:) = [];
re_dets.cluster_locs(:,11) =1:length(r);

rr = re_dets.cluster_locs(:,12);
[~,ir] = sort(rr,'descend');
re_dets.cluster_locs = re_dets.cluster_locs(ir,:);

toc
% 
% conf.clustering.top_k = inf;
% phase1_scores = cur_locs(:,12);
% alpha_ = 1
% disp(alpha_)
% re_topdets = re_dets;
% % add to all of them alpha*phase 1 grades.
% 
% for k = 1:length(re_topdets)
%     re_topdets(k).cluster_locs(:,12) = re_topdets(k).cluster_locs(:,12) + ...
%         alpha_*phase1_scores(re_topdets(k).cluster_locs(:,11));
%     
%     [~,ia] = sort(re_topdets(k).cluster_locs(:,12),'descend');
%     re_topdets(k).cluster_locs = re_topdets(k).cluster_locs(ia,:);
% end


% if (nargin >= 8) % use betas
%     if (~isempty(betas))
%         use_best = 0;
%         for k = 1:length(re_topdets)
%             fx = @(x)(1./(1+exp(-betas(k,1)*(x-beta(k,2)))));
%             re_topdets(k).cluster_locs(:,12) = fx(re_topdets(k).cluster_locs(:,12));
%         end
%     else
%          use_best = 1;
%     end    
% end

%
gt_labels_ = gt_labels(cur_locs(:,11));
[~,~,aps] = calc_aps(re_dets,gt_labels_,[],inf);
[m,im] = sort(aps,'descend');
if (nargin <9)
    iim = im;
end

im = iim;
disp(max(m))

c = 1;
% im=  1:8;
d = re_dets(im(c)).cluster_locs(:,11);
num_show = 25;
d = d(1:min(size(d,1),min(k_topdets,num_show)));
re_topdets = re_dets;
if (toshow)
    close all;
    draw_rect = false;
    [p] = visualizeLocs2(conf,r,re_dets(im(c)).cluster_locs,64,1,0,draw_rect);
    figure,imshow(multiImage(p(1:min(length(p),num_show))));
    visPath_vis =  fullfile(conf.cachedir,['phase2_vis4vis' suffix '_' num2str(k_topdets) '.mat']);
    if (~exist(visPath_vis,'file'))
        r_ = visualizeLocs2(conf,train_set,cur_locs,64,inflate_factor,1);
        save(visPath_vis,'r_');
    else
        load(visPath_vis);
    end
    
    ff = multiImage(r_(d));
    true_img = multiImage(r_(1:size(d,1)));
    figure,imshow(true_img);title('phase 1');
    figure,imshow(ff);title('phase 2');
    outPath1 = fullfile('res_show',[suffix '_phase1.jpg']);
    outPath2 = fullfile('res_show',[suffix '_phase2.jpg']);
    % also visualize the actual detections by this cluster...
    
    imwrite(true_img,outPath1,'Quality',100);
    imwrite(ff,outPath2,'Quality',100);
    imwrite(imresize(multiImage((p(1:min(length(p),num_show)))),[size(ff,1),size(ff,2)],'bilinear'),...
        strrep(outPath2,'.jpg','_exp.jpg'),'Quality',100);
end
cur_new = [];

if (use_best)
    
    for k = 1%1:length(re_topdets)
        dd =inf;
        ii =im(k);
        nlocs = size(re_topdets(ii).cluster_locs,1);
        cur_new = [cur_new;re_topdets(ii).cluster_locs(1:min(nlocs,dd),:)];
    end
else % unite all detections.
    for k = 1:length(re_topdets)
        dd =inf;
        ii = k;
        nlocs = size(re_topdets(ii).cluster_locs,1);
        cur_new = [cur_new;re_topdets(ii).cluster_locs(1:min(nlocs,dd),:)];
    end
    
end

%%
s_ = size(cur_new,1);
dets.cluster_locs(1:s_,11) = cur_locs(cur_new(:,11),11);
dets.cluster_locs(1:s_,12) = cur_new(:,12);
% dets.cluster_locs = dets.cluster_locs(1:s_,:);
%dets.cluster_locs(s_+1:end,12) = min(dets.cluster_locs(s_+1:end,12),min(dets.cluster_locs(1:s_,12)));
dets.cluster_locs(s_+1:end,12) = dets.cluster_locs(s_+1:end,12) -...
    (max(dets.cluster_locs(s_+1:end,12))-min(dets.cluster_locs(1:s_,12)));

% rearrange re_topdets so indices relate to oroginal ones.
for k = 1:length(re_topdets)
    re_topdets(k).cluster_locs(:,11) = cur_locs(re_topdets(k).cluster_locs(:,11),11);
end

%%
% % %     cur_new = [];
% % %     tt = 2;
% % %
% % %     for q = 1:tt
% % %         for k = 1:length(re_topdets)
% % %             ii =k%im(k)
% % %             p = re_topdets(ii).cluster_locs(q,:);
% % %             p(:,12) = 2*tt-q;
% % %             cur_new = [cur_new; p];
% % %         end
% % %     end
% % %
% % %     for k = 1:length(re_topdets)
% % %         ii =k%im(k)
% % %         p = re_topdets(ii).cluster_locs(tt+1:end,:);
% % %         cur_new = [cur_new; p];
% % %     end
