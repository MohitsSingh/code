function [w_poi, b_poi] = train_poi_detector(conf,fra_db,f_train,params,params_coarse,featureExtractor)
params.cand_mode = 'boxes';
% cur_set = f_train_pos;
roi_pos_patches = {};
roi_neg_patches = {};
nodes = params.nodes;
dlib_landmark_split;
w_int = params_coarse.w_int;
b_int = params_coarse.b_int;
w = params.w;
b = params.b;
cur_set = f_train;
nSamples = 0;
samples = struct('label',{},'img',{},'masks',{});

nSamples = 0;
tic_id = ticStatus('generating fine samples...',.5,.5);

for it = 1:length(cur_set)
    t = cur_set(it);
    imgData = fra_db(t);
    %     [applyDetector(conf,imgData,
    I = getImage(conf,imgData);
    % 1.Detect coarse region of interaction
    [I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData);
    mouthMask = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib);
    [rois,scores,box_feats] = detect_coarse(conf,imgData,params_coarse,featureExtractor,w_int,b_int);
    [r,ir] = max(scores);
    coarse_bb = rois(ir,:);
    p0 = boxCenters(mouthBox);
    % sample points inside the best roi
    p1 = sampleGridInBox(coarse_bb,5);
    
    % 2. generate candidate regions within this one
    %     displayRegions(I,regions,scores)
    %     displayRegions(I,regions);
    rois1 = {};
    for ip = 1:size(p1,1)
        rois1{ip} = directionalROI_rect_lite(p0,p1(ip,:),.2*size(I_sub,1));
    end
    masks = cellfun2(@(x) poly2mask2(x,size2(I)),rois1);
    
    % rank the masks according to the ground-truth region overlap, if any
    
    %     patches = cellfun2(@(x) maskedPatch(I,x,true,.5),masks);
    [scores,coarse_feats] = scoreRegions(I,masks,featureExtractor,w_int,'mask_patches',true);
    
    % 3. find segmentation dependent regions overlapping with the generated
    % regions and score them.
    [candidates,isvalid] = getCandidateRegions(conf,imgData);
    masks_local = cellfun2(@(x) cropper(x, mouthBox),masks);
    %     masks_local = removeDuplicateRegions(masks_local);
    regions = processRegions(I_sub,candidates,mouthMask); % remove some bad regions
    [ovps ints uns] = regionsOverlap3(regions,masks_local);
    [m,im] = max(ovps,[],2);
    masks1 = regions(m>.3); % remove regions which do not overlap with the hypothesized masks enough.
    regions = [regions,masks1];
    regions = removeDuplicateRegions(regions);
    if  params.isClass(t) % get some ground truth regions
        gt_graph = get_gt_graph(imgData,nodes,params,I);
        % 2. remove "trivial" regions
        if ~isfield(gt_graph{2},'roiMask')
            continue
        end
        m = gt_graph{2}.roiMask;
        if isempty(m) || nnz(m)==0
            continue
        end
        gtMask = cropper(m,mouthBox);
        if (nnz(gtMask)==0)
            continue
        end
        if strcmp(params.phase,'training')
            regions = [regions,gtMask];
        end
        [ovp,ints,uns] = regionsOverlap(regions,gtMask);
        
        sel_ = ovp > .5;
        ovp = ovp(sel_);
        regions = regions(sel_);
        [r,ir] = sort(ovp,'descend');
        regions = regions(ir(1:min(length(regions),10)));
        label = 1;
        %bestRoi = rois{ir(1)};
    else
        label = -1;
        regions = vl_colsubset(regions,10);
    end
    nSamples = nSamples+1;
    samples(nSamples).label = label;
    samples(nSamples).img = I_sub;
    samples(nSamples).masks = regions;
    
    
    %     [scores,feats] = scoreRegions(I,regions,featureExtractor,w,'mask_patches',true);
    %     clf; imagesc2(I); plotBoxes(coarse_bb); plotPolygons(p1,'r.');
    %     displayRegions(I,masks,scores,0,1);
    tocStatus(tic_id,it/length(cur_set));
end

[feats,labels] = samplesToFeats(samples,featureExtractor);
classifier_data = Pegasos(feats,labels(:),'lambda',.0001);%,...
w_poi = classifier_data.w(1:end-1);
b_poi = classifier_data.w(1:end-1);


end

