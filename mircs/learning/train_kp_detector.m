%function train_kp_detector(face_data_t)
addpath(genpath('/home/amirro/code/3rdparty/svm-struct/'));    
% setup:

% function doStuff()
randn('state',0);
rand('state',0);

% model_types = {'nolearn','minimal','partial','full','full_edges'};
cellSize = 3;
windowSize = cellSize*8;
loc_fraction = 1;
imgSize = 64;
iExp = 0;
doBenchmark = true;
forceTrain = true;
param.keepDeadStates = true;
param.use_pairwise_scores = true;
param.use_location_prior = false;
isTreeStructure = true;

data_train = face_data_t(1:2000);

% % % [phisTr,IsTr] = prepareData2(data_train);
% % % 
% % % phisT = phisTr(1:2:end,:,:);
% % % IsT = IsTr(1:2:end);
% % % phisTr = phisTr(2:2:end,:,:);
% % % IsTr = IsTr(2:2:end);
%%
debug_factor = 5;

learningType = 'nolearn';

kp_sel_bit = zeros(1,21);
% kp_sel = 1:21;
kp_sel = [7 8 20 21 12 10 9 11 1];
% kp_sel = [6 19 12 10 11 1]; % eye centers, tip of nose, mouth corners, chin
 % eye corners, tip of  nose, left-right-center
% mouth, chin
kp_sel_bit(kp_sel) = 1;
kp_sel_str = strrep(num2str(kp_sel_bit),' ','');
imgSize_c = imgSize/cellSize;
windowSize_c = windowSize/cellSize;
[ phis_tr_n ,factors] = normalize_coordinates( phisTr,IsTr,false);
IsTr_1 = multiResize(IsTr,imgSize);
IsTr_1 = cellfun2(@(x) condition(length(size(x))==3,x,cat(3,x,x,x)),IsTr_1);
IsT_1 = multiResize(IsT,imgSize);
IsT_1 = cellfun2(@(x) condition(length(size(x))==3,x,cat(3,x,x,x)),IsT_1);
phis_tr_n = phis_tr_n(:,:,1:2);
patterns = IsTr_1;
nPts = length(kp_sel);
labels = {};
for t = 1:length(patterns)
    labels{t} = reshape(phis_tr_n(t,kp_sel,:),[],2);
end

% remove examples where there are NaN values
goods = find(cellfun(@(x) none(isnan(col(x))),labels));
patterns = patterns(goods);
labels = labels(goods);


nPts = length(kp_sel);

infer_visibility = false;

% location prior: create a tree from the keypoint structure.
% since all images are of the same size, can already model the prior & pairwise
% constraints here.
boxes = get_boxes_single_scale(windowSize, cellSize,zeros(imgSize,imgSize,3,'single'),'dense');
all_locs = boxCenters(boxes)/imgSize;
% generate the pairwise constraints for all pairs in the tree.
nLocs = size(all_locs,1);
param.nLocsAdmitted = round(loc_fraction*nLocs);
all_kp_locs = cat(3,labels{:});
toSnap = false;
if (toSnap)
    % snap to grid
    for iLoc = 1:size(all_kp_locs,3)
        curLocs = all_kp_locs(:,1:2,iLoc);
        d = l2(curLocs,all_locs);
        [m,im] = min(d,[],2);
        all_kp_locs(:,1:2,iLoc) = all_locs(im,:);
        labels{iLoc}(:,1:2) = all_locs(im,:);
    end
end
% all_kp_visible = all_kp_locs(:,3,:);
n_gaussian_components = 1;
gaussian_models = {};
% figure(1); clf;
% hold on;
loc_priors = zeros(nLocs,nPts);

hard_loc_priors = zeros(nLocs,nPts);

for t = 1:nPts
    GMModel = fitgmdist(squeeze(all_kp_locs(t,1:2,:))',n_gaussian_components);%,'CovType','Diagonal');
    gaussian_models{t} = GMModel;
    loc_priors(:,t) = log_lh(GMModel,all_locs);
    %ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}));
    
    
    
    %     ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),4);
    %     pause
end

% model a pairwise term between patches, using distances.

Q = cat(4,IsTr_1{:});
Q = mean(im2single(Q),4);

% remove states that cannot happen : find the top-20 percent of
% each prior.

[z,iz] = sort(loc_priors,'descend');
loc_admitted = false(size(loc_priors));
for t = 1:nPts
    loc_admitted(iz(1:param.nLocsAdmitted,t),t) = true;
end


yhats = zeros(nPts,param.nLocsAdmitted,2);
for t = 1:nPts
    yhats(t,:,:) = all_locs(loc_admitted(:,t),:);
end
%             for zz = 1:length(IsTr)
%             clf;imagesc2(IsTr_1{zz});plotPolygons(imgSize*squeeze(yhats(1,:,:)),'r.')
%             drawnow;pause;
%             end
param.yhats = yhats;
% binarize it...
param.loc_admitted = loc_admitted;
%                 param.loc_prior = log(loc_priors);
%             admitted_subsets = loc_priors > 1;

gaussian_means = cellfun2(@(x) x.mu,gaussian_models);gaussian_means = cat(1,gaussian_means{:});
G = l2(gaussian_means,gaussian_means).^.5;

param.isTreeStructure = isTreeStructure;
% make sure that only i > j entries are non zeros in adj.
% matrix!!
if isTreeStructure
    adj = G;
    [adj,pred] = graphminspantree(sparse(adj),'Method','Kruskal');
    %                     adj = adj';
else
    %                 % make this a sparser model by removing long edges
    G(G>.31) = 0;
    for i = 1:nPts
        for j = i:nPts
            G(i,j) = 0;
        end
    end
    
    adj = sparse(G);
end
%                 adj = adj;
%                 for img_sel = 1:40:400


img_sel = find(cellfun(@(x) none(isnan(col(x))),labels),1,'first');

curImg = patterns{img_sel};
figure(10);clf ; imagesc2(curImg);
sz = size(curImg,1);
curCoords = sz*labels{img_sel}(:,1:2);
%                 adj(3,1) = 0;adj(4,2) = 0;

gplot2(adj,squeeze(sz*labels{img_sel}(:,1:2)),'g-','LineWidth',2);
showCoords(curCoords);
title('graph')
%                 pause
                                plotBoxes(inflatebbox(repmat(curCoords,1,2),windowSize,'both',true));
%                                 pause;
%                 end
adj = double(adj > 0);
[ii,jj] = find(adj);
edges = [ii jj];
nEdges = length(ii);

%% define a pairwise score by a 2d-gaussian for the relative location

% of each pair of keypoints.
if loc_fraction == 1 || param.keepDeadStates
    pairwise_scores = -inf(nLocs,nLocs,nEdges);
else
    pairwise_scores = -inf(param.nLocsAdmitted,param.nLocsAdmitted,nEdges);
end
%TODO: LOCADMIT
%
pair_gaussian_models = cell(nEdges,1);
[loc_i,loc_j] = meshgrid(1:nLocs,1:nLocs);
zz = all_locs(loc_j(:),:)-all_locs(loc_i(:),:);
for iEdge = 1:nEdges
    %                     iEdge
    iPair = edges(iEdge,1);
    jPair = edges(iEdge,2);
    
    offsets = squeeze(all_kp_locs(jPair,1:2,:)-all_kp_locs(iPair,1:2,:))';
    pair_gaussian_models{iEdge} = fitgmdist(offsets,n_gaussian_components);%,'CovType','Diagonal');
    %                     disp(edges(iEdge,:))
    %                     pair_gaussian_models{iEdge}
    % %     clf;
    % %     ezcontour(@(u,v)pdf(pair_gaussian_models{iEdge},[u v]),300)
    % %     hold on; plot(0,0,'r+');
    % %     pause
    p = log_lh(pair_gaussian_models{iEdge},zz);
    %figure,imagesc2(IsTr_1{1})
    p = reshape(p,nLocs,nLocs);
    loc_subset_i =  loc_admitted(:,iPair);
    loc_subset_j =  loc_admitted(:,jPair);
    % TODO: LOCADMIT
    if (loc_fraction==1 || param.keepDeadStates)
        pairwise_scores(:,:,iEdge) = p;
    else
        pairwise_scores(:,:,iEdge) = p(loc_subset_i,loc_subset_j); %TODO?
    end
    
    visualize_stuff = false;
    if (visualize_stuff)
        [xx,yy] = meshgrid((1:imgSize)/imgSize,(1:imgSize)/imgSize);
        curMean = gaussian_models{iPair}.mu;
        z = reshape(log_lh(pair_gaussian_models{iEdge},[xx(:)-curMean(1) yy(:)-curMean(2)]),size(xx));
        prob_image = exp(z/2);
        clf ; imagesc2(sc(cat(3,prob_image,curImg),'prob'));
        sz = size(curImg,1);
        curCoords = sz*labels{img_sel}(:,1:2);
        plotPolygons(curMean*imgSize,'r+');
        gplot2(adj,squeeze(sz*labels{img_sel}(:,1:2)),'g-','LineWidth',2);
        showCoords(curCoords);
        title(num2str(edges(iEdge,:)));
        %                         continue
        % do the same with the image locs: find the pairwise
        % score relating to a specific location for i and read
        % off the probabilities for the location of j
        
        % find the nearest neighbor to the eye corner
        if (0)
            clf ;imagesc2(curImg); [x1,y1] = ginput(1);
            [u,iu] = min(l2([x1,y1]/imgSize,all_locs))
            all_locs(iu,:)
            log_probs = pairwise_scores(iu,:,iEdge);
            m = visualizeTerm(log_probs',all_locs*imgSize,[imgSize imgSize]);
            figure(1);imagesc2(exp(m));figure(2);imagesc2(curImg);
        end
        %                     figure,imagesc2(exp(z/2)); figure,imagesc2(IsTr_1{1})
    end
    %
end

%                 pairwise_scores(pairwise_scores==-inf) = min(pairwise_scores(pairwise_scores(:)>-inf));
param.pairwise_scores = pairwise_scores;
param.nEdges = nEdges;
param.edges = edges;
param.patterns = row(patterns(1:debug_factor:end));
param.pair_gaussian_models = pair_gaussian_models;
param.gaussian_models = gaussian_models;
param.labels = row(labels(1:debug_factor:end));


param.lossFn = @lossCB;
param.lossType = 2;
param.adj = adj;
switch learningType
    case {'minimal','nolearn'}
        param.constraintFn  = @constraintCB_minimal;
        param.featureFn = @featureCB_minimal;
        wDim = 3;  % wvisible+winvisible+w_app
    case {'minimal_occ','nolearn_occ'}
        param.constraintFn  = @constraintCB_minimal_occ;
        param.featureFn = @featureCB_minimal_occ;
        wDim = 4;  % wvisible+winvisible+w_app
    case 'partial'
        param.constraintFn  = @constraintCB;
        param.featureFn = @featureCB;
        wDim = nPts+nEdges;
    case 'full'
        param.constraintFn  = @constraintCB_full;
        param.featureFn = @featureCB_full;
        wDim = (31*(windowSize/cellSize)^2)*nPts + nEdges;
    case 'full_edges'
        param.constraintFn  = @constraintCB_full_edges;
        param.featureFn = @featureCB_full_edges;
        wDim = (31*(windowSize/cellSize)^2)*nPts + nEdges*5;
end

param.dimension = wDim;
% param.dimension = 2;

param.verbose = 1;
param.imgSize = imgSize;
param.windowSize = windowSize;
param.nPts = nPts;
param.cellSize = cellSize;
param.debug = false;

if (~param.use_location_prior)
    param.location_priors = zeros(size(loc_priors));
else
    param.location_priors = loc_priors;
end
param.use_appearance_feats = true;
param.infer_visibility = infer_visibility;
%                 param.rotations = -20:20:20;
param.rotations = 0;

% % %% first model : learn a random field for the locations.
negImagePaths = getNonPersonImageList();
%
% save z z
% train patch detectors

detector_path = sprintf('detectors_new_img_%d_w_%d_c_%d.mat',imgSize,windowSize,cellSize);
if (~exist(detector_path,'file'))
    [~,param.patchDetectors] = learnPatchModels(param,negImagePaths);
    z = param.patchDetectors;
    save(detector_path,'z');
else
    load(detector_path);
end
param.patchDetectors = z(kp_sel,:);
%             load z.mat;

param.phase = 'train';
%param.precompute_responses = true && strcmp(learningType,'partial');
param.precompute_responses = false && strcmp(learningType,'partial');
if (param.precompute_responses)
    param.patterns = precompute_responses(param,param.patterns);
end

%%
if strcmp(learningType,'nolearn')
    model.w = [1 1 1];
else
    % use structured learning
    % profile off
    
    param.use_pairwise_scores = 1;
    param.use_appearance_feats = 1;
    
    %                     profile on
    param.phase = 'train';
    %             profile on
    model = svm_struct_learn(' -c .1 -o 2 -v 2 -w 4', param) ;
    %                         profile viewer
end
%%
test_subset = 1;
nTest = length(IsT);
tt = 1:test_subset:nTest;
%                 tt =191
%                 param.rotations = [0 -15 -30];
%
param.rotations = 0;
%                 profile off
%                 desiredLocs = [20 18;46 17; 33 44];
%                 param.pairwise_scores = make_dummy_pairwise_scores(param,desiredLocs);
param.rotations = 0;
uu = 0;
if uu==0
    param.rotations = 0;
else
    param.rotations =-uu:uu:uu;
end
param.toFlip = true;
%                 model.w = [1 .1];
param.phase = 'test';
[gts,preds,stats] = apply_to_imageset(IsT(tt),phisT(tt,:,:),model,...
    param,kp_sel,learningType,true);

%%
% % % % load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
% % % % s40_fra = s40_fra_faces_d;
% % % % %%
% % % % 
% % % % 
% % % % IsT_s40 = {};
% % % % for t = 2000:1:length(s40_fra)
% % % %     t/9532
% % % %     if (s40_fra(t).classID ~= conf.class_enum.DRINKING)
% % % %         continue
% % % %     end
% % % %     face_box = round(inflatebbox(s40_fra(t).faceBox,1.3,'both',false));
% % % % %     clf; imagesc2(I); plotBoxes(face_box); 
% % % % if (s40_fra(t).faceScore>0)
% % % %     I = getImage(conf,s40_fra(t));    
% % % %     IsT_s40{end+1} = cropper(I,face_box);
% % % %     t
% % % % end        
% % % % %     drawnow;
% % % % %     pause
% % % % end

%%
[gts,preds,stats] = apply_to_imageset(IsT_s40,[],model,...
    param,kp_sel,learningType,true);

% x2(IsT_s40);

% apply to some "real" images
