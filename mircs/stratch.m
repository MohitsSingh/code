%% look at the simple union of all detections...
calc_united_ap = 1;
if (calc_united_ap)
    option_ = 1;
    % [h,ih] = sort(hit_rate,'descend');
    r = {};
    
    % calculate the maximal score per image of each classifier.
    
    M = -2*ones(length(gt_labels),length(top_dets));
    IM = zeros(size(M));
    
    rrr = ibest;
    for k = 1:length(top_dets)
        k
        scores = top_dets(rrr(k)).cluster_locs(:,12);
        inds = top_dets(rrr(k)).cluster_locs(:,11);
        z = -2*ones(size(gt_labels))';
        for iInd = 1:length(inds)
            [z(inds(iInd)),im] = max([z(inds(iInd)),scores(iInd)]);
            if (im ==2)
                IM(inds(iInd),k) = iInd;
            end
        end
        M(:,k) = z;
    end
    
    %%
    for qq = 1:length(top_dets)
        % % %     close all;
        
        all_locs = {};
        
        scores = -inf*ones(1000,1);
        locs_ = [];
        
        % rrr = randperm(50);
        %rrr = ibest;
        
        %     rrr = 1:50;
        [z,izz] = max(M(:,1:qq),[],2);
        
        npos = sum(gt_labels);
        rec = tp/npos;
        prec = tp./(fp+tp);
        
        ap = VOCap(rec,prec)
        r{qq} = ap;
        % % % figure,plot(rec,prec)
        
        % % locs_1 = cat(1,locs_{iz(1:100)});
        % % vvv= visualizeLocs(conf,test_set,locs_1,64);
        % % V_ = multiImage(vvv);
        % % figure,imshow(V_); title(ap);
        % pause;
    end
    
end

%%
% % % addpath('/home/amirro/code/3rdparty/face-release1.0-basic');
% % % %load face_p146_small.mat
% % %
% % % [ids,labels] = getImageSet(conf,'train');
% % % I = toImage(conf,getImagePath(conf,ids{1}));
% % % figure,imshow(I)
% % %
% % %  im = I;
% % %     clf; imagesc(im); axis image; axis off; drawnow;
% % %
% % %     tic;
% % %     bs = detect(im, model, model.thresh);
% % %     bs = clipboxes(im, bs);
% % %     bs = nms_face(bs,0.3);
% % %     dettime = toc;
% % %
% % %
% % %     % show all
% % %     figure,showboxes(im, bs),title('All detections above the threshold');
% % %
% % %

%%
initpath;
config;
addpath(genpath('/home/amirro/code/3rdparty/face-release1.0-basic'));
load multipie_independent.mat
model.thresh = min(-.65, model.thresh);
model.interval = 5;

[ids,labels] = getImageSet(conf,'train');
tic;
f = find(~labels);
conf.max_image_size = 512;
%%
close all;
sel = 3;
im = toImage(conf,getImagePath(conf,ids{f(sel)}));
im = im2uint8(im);
im = imcrop(im);
im = imresize(im,2,'bicubic');
figure,imshow(im)

bs = detect(im, model, model.thresh);
bs = clipboxes(im, bs);
bs = nms_face(bs,0.3);
dettime = toc;

% show highest scoring one
%     figure,showboxes(im, bs(1),posemap),title('Highest scoring detection');
% show all
figure,showboxes(im, bs,posemap),title('All detections above the threshold');


%%
%%
addpath(genpath('/home/amirro/code/3rdparty/proposals'));
f = find(labels);
conf.max_image_size = 128;
I = toImage(conf,getImagePath(conf,ids{f(1)}));
[ranked_regions superpixels image_data] = generate_proposals(I);

masks = {};

for kk = 1:length(ranked_regions)
    kk/length(ranked_regions);
     mask = ismember(superpixels, ranked_regions{kk});
%      masks{kk} = mask;
     mask = cat(3,mask,mask,mask);
     
     imshow(im2double(mask).*im2double(I));
     pause;
end
%%
