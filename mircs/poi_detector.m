ffunction [w_poi, b_poi,samples] = poi_detector(conf,fra_db,cur_set,params,params_coarse,featureExtractor,sample_stats,normalizer)
params.cand_mode = 'boxes';
% cur_set = f_train_pos;
roi_pos_patches = {};
roi_neg_patches = {};
nodes = params.nodes;
dlib_landmark_split;
w_int = params_coarse.w_int;
b_int = params_coarse.b_int;
% w = params.w;
% b = params.b;
nSamples = 0;
samples = struct('label',{},'img',{},'masks',{},'coarse_score',{},'imgInd',{});
isTrainingPhase = strcmp(params.phase,'training');
nSamples = 0;
tic_id = ticStatus('generating fine samples...',.5,.5);
rlfe = RelativeLayoutFeatureExtractor_2(conf);

% 
% for it = 1:length(cur_set)
%     it/length(cur_set)
%     t = cur_set(it);
%     imgData = fra_db(t);
% %     if imgData.classID~=4,continue,end
%     I = getImage(conf,imgData);
%     % 1.Detect coarse region of interaction
%     [I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData);
%     [candidates,ucm2,isvalid] = getCandidateRegions(conf,imgData,I_sub);
% end

for it = 1:length(cur_set)
    t = cur_set(it);
    imgData = fra_db(t);
%     if imgData.classID~=4,continue,end
    I = getImage(conf,imgData);
    % 1.Detect coarse region of interaction
    [I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData);
    [mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
    [rois,scores,box_feats] = detect_coarse(conf,imgData,params_coarse,featureExtractor,w_int,b_int);
    [r,ir] = max(scores);
    coarse_score = r;
    coarse_bb = rois(ir,:);
    p0 = boxCenters(mouthBox);
    
    
    %%
    x2(I_sub);
    plot_dlib_landmarks(curLandmarks);
    %%
    
    %% find if there is a face occluder in this region
    % % %     [candidates_g,ucm_g,success_g] = getRegions(conf,imgData);
    % % %
    % % %     ucm_g = (ucm_g(1:2:end,1:2:end));
    % % %     figure(1);clf; imagesc2(ucm_g.^.5);
    % % %     plotBoxes(coarse_bb);
    % % %     figure(2); imagesc2(I);
    % % %     plotBoxes(coarse_bb);
    % % %
    % % %     E = edge(im2double(rgb2gray(I)),'canny');
    % % %
    % % %     x2(ucm_g>.2)
    % % %     x2(I)
    
    %%    
    %sample points inside the best roi    
    [candidates,ucm2,isvalid] = getCandidateRegions(conf,imgData,I_sub);
    %
    %     figure(1); clf; subplot(1,2,1); imagesc2(I_sub);
    %     subplot(1,2,2);imagesc2(ucm2(1:2:end,1:2:end));
    %     I_sub = imResample(I_sub,2);
    %     [candidates, ucm2]  = im2mcg(I_sub,'fast');
    %     figure(2); clf; subplot(1,2,1); imagesc2(I_sub);
    %     subplot(1,2,2);imagesc2(ucm2(1:2:end,1:2:end));
    
    %     continue
    %dpc; continue
    if (~isvalid)
        %         disp('skipping invalid sample...');
        continue
    end
    
    if (size(I_sub,1) < 25)
        continue
    end
    
    %     mouthMask = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib);
    regions = processRegions(I_sub,candidates,mouthMask); % remove some bad regions
    %regionFeats = extractFeaturesForSegment(img,mask,featureExtractor,debug_,params)
    %     regionFeats = extractFeaturesForSegment(I_sub,regions{1},featureExtractor);
    [regions_parametric] = generateParametricRegions(I,p0,I_sub,coarse_bb,mouthBox);
    [ovps ints uns] = regionsOverlap3(regions,regions_parametric);
    [m,im] = max(ovps,[],2);
    masks1 = regions(m>.3); % remove regions which do not overlap with the hypothesized masks enough.
    regions = [masks1,regions_parametric];
    regions = removeDuplicateRegions(regions);
    regionBoxes = cellfun3(@region2Box,regions);
    regionBoxes_rel_mouth = bsxfun(@minus,regionBoxes,region2Box(mouthMask));
    regionBoxes_rel_mouth = regionBoxes_rel_mouth/size(I_sub,1);
    px = regionBoxes_rel_mouth(:,1).*regionBoxes_rel_mouth(:,3);
    py = regionBoxes_rel_mouth(:,2).*regionBoxes_rel_mouth(:,4);
    regions(px < 0 & py < 0) = [];
    centers = cellfun3(@(x) mean(region2Pts(x)),regions);
    I_sub_t = I_sub; for u = 1:3,rr = I_sub_t(:,:,u); rr(mouthMask) = 1; I_sub_t(:,:,u) = rr; end
    % do geometric processing...
    % apply rules of interaction.
    % get both "tips" of each region.            
    [R,rects,direction_vecs,to_mouth_vecs] = getRectData(regions,I_sub,curLandmarks,mouthMask);
    direction_agreement = abs(sum(to_mouth_vecs.*direction_vecs,2));
    R(direction_agreement<.5) = [];
    rects(direction_agreement<.5) = [];
    regions(direction_agreement<.5) = [];
    R = cat(1,R{:});
    y = sample_stats.pdf(R);
    [z,iz] = sort(y,'descend');
    ovps = regionsOverlap3(regions,regions);
    S = suppresRegions(ovps,.8,y,I,regions);
    S = S{1};
    regions = regions(S);
    y = y(S);
    R = R(S,:);
    rects = rects(S);
    px = px(S);
    py = py(S);
    geom_descs = getGeomFeats(curLandmarks,mouthMask,regions);
    assert(size(geom_descs,1)==52)
    % get some more geometric info....
    if  params.isClass(t) && isTrainingPhase % get some ground truth regions
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
        regions = [regions,gtMask];
                
        [R_gt,rects_gt,direction_vecs_gt,to_mouth_vecs_gt] = getRectData({gtMask},I_sub,curLandmarks,mouthMask);        
        geom_descs = [geom_descs,getGeomFeats(curLandmarks,mouthMask,{gtMask})];
        [ovp,ints,uns] = regionsOverlap(regions,gtMask);
        sel_ = ovp > .5;
        ovp = ovp(sel_);
        regions = regions(sel_);
        geom_descs = geom_descs(:,sel_);        
        
        
        R(end+1,:) = zeros(size(R(1,:)));
        rects{end+1} = rects{end};
        R = R(sel_,:);
        rects = rects(sel_);
        [r,ir] = sort(ovp,'descend');
        sel_ = ir(1:min(length(regions),10));
        regions = regions(sel_);
        geom_descs = geom_descs(:,sel_);
        R = R(sel_,:);
        rects = rects(sel_);
    else
        if isTrainingPhase
            sel_ = vl_colsubset(1:length(regions),10,'Uniform');
            regions = regions(sel_);
            geom_descs = geom_descs(:,sel_);
            R = R(sel_,:);
            rects = rects(sel_);
        end
    end
    
    if params.isClass(t)
        label = 1;
    else
        label = -1;
    end
    
    nSamples = nSamples+1;
    
    samples(nSamples).label = label;
    samples(nSamples).img = I_sub;
    samples(nSamples).masks = regions;
    samples(nSamples).geom_descs = geom_descs;
    
    %     R = regions{15};
    %     R = imerode(R,ones(11));
    %     [gc_segResult2,obj_box] = checkSegmentation(I_sub,R);
    
    if ~isTrainingPhase
        %[ scores,feats ] = scoreRegions(I_sub,samples(nSamples),...
        %    featureExtractor,params.w_poi,params.b_poi,'normalizer',normalizer);
        
        [ scores,feats ] = scoreRegions(I_sub,samples(nSamples),...
           featureExtractor,params.w_poi,params.b_poi);
        [r,ir] = sort(scores,'descend');
        %         ir = ir(1:min(length(ir),10));
        %         scores = scores(ir);
        samples(nSamples).masks = regions(ir);
        samples(nSamples).scores = r;
%         geom_descs = geom_descs(:,ir);
    end
    
%     for iRegion = 1:length(regions)
%         clf; displayRegions(I_sub,regions(iRegion),[],'dontPause',true);
%         plotPolygons(rects{iRegion},'g-');
%         plotPolygons(size(I_sub,1)*R(iRegion,1:2),'m*');
%         dpc;
%     end
    
    samples(nSamples).mouthMask = mouthMask;
    coarse_bb_local = coarse_bb-faceBox([1 2 1 2]);
    samples(nSamples).coarse_bb_local = coarse_bb_local;
    samples(nSamples).coarse_score = coarse_score;
    samples(nSamples).imgInd = t;
    samples(nSamples).geom_descs = geom_descs;
    
    assert(size(geom_descs,1)==52);
    assert(none(isnan(geom_descs(:))))
    
    
    tocStatus(tic_id,it/length(cur_set));
end
if isTrainingPhase
%     save samples_tmp.mat samples
    [feats,labels] = samplesToFeats(samples,featureExtractor);
    
%     feats = [feats;geom_feats];
    classifier_data = Pegasos(feats,labels(:),'lambda',.0001);%,...
    w_poi = classifier_data.w(1:end-1);
    b_poi = classifier_data.w(1:end-1);
else
    w_poi = params.w_poi;
    b_poi = params.b_poi;
end


function [regions_parametric] = generateParametricRegions(I,p0,I_sub,coarse_bb,mouthBox)
p1 = sampleGridInBox(coarse_bb,5);
% 2. generate candidate regions within this one
%     displayRegions(I,regions,scores)
%     displayRegions(I,regions);
rois1 = {};
for ip = 1:size(p1,1)
    rois1{ip} = directionalROI_rect_lite(p0,p1(ip,:),.2*size(I_sub,1));
end
regions_parametric = cellfun2(@(x) poly2mask2(x,size2(I)),rois1);

regions_parametric = cellfun2(@(x) cropper(x, mouthBox),regions_parametric);

[regions_parametric,sel_] = ezRemove(regions_parametric,I_sub,50,.3);
regions_parametric = row(regions_parametric);
% polys = rois1(sel_);
bb_pts = box2Pts(mouthBox);
% polys = cellfun2(@(x) polybool2('&',x,bb_pts), polys);

function [R,rects,direction_vecs,to_mouth_vecs] = getRectData(regions,I_sub,curLandmarks,mouthMask)

rects = {};
R = {};

direction_vecs = {};
to_mouth_vecs = {};
%ss = size2(I_sub)/2;
mouthPts = region2Pts(mouthMask);
mouthCenter = mean(mouthPts);

%%
showStuff = false;
for ir = 1:length(regions)
    [y,x] = find(regions{ir});
    [rx,ry] = minboundrect(x,y);
    curRect = [rx(1:4) ry(1:4)];
    m = mean(curRect);
    rects{ir} = curRect;
    diffs = diff(curRect);
    centers = (curRect+curRect([2:end 1],:))/2;
    lengths = sum(diffs.^2,2).^.5;
    [w,iw] = min(lengths);
    
    p0 = centers(iw,:);
    iw = iw+2;
    if iw > 4
        iw = 1;
    end
    p1 = centers(iw,:);
    dist_to_mouth = l2([p0;p1],mouthCenter);
    
    if dist_to_mouth(1) > dist_to_mouth(2) % swap so p0 is closer to mouth.
        [p0,p1] = deal(p1,p0);
    end
    
    dVec = p0-p1;
    
    direction_vecs{end+1} = dVec;
    toMouthVec = mouthCenter-m;
    to_mouth_vecs{end+1} = toMouthVec;
    
    R{ir} = [p0 p1 w]/size(I_sub,1);
    
    dist_mouth_mask = l2([p0;p1],mouthPts);
    [d,id] = min(dist_mouth_mask,[],2);
    
    %     mouthCorners = curLandmarks([49,55],:);
    %     d1 = l2(p0,mouthCorners);
    %     d2 = l2(p1,mouthCorners);
    v1 = -(p0-mouthPts(id(1),:));
    v2 = -(p1-mouthPts(id(2),:));
    if showStuff
        
        clf;displayRegions(I_sub,regions{ir},[],'dontPause',true); plotPolygons(curRect,'g-');
        plot_dlib_landmarks(curLandmarks+1)
        plotPolygons(p0,'ys','LineWidth',5);
        plotPolygons(p1,'m*')
        quiver(m(1),m(2),dVec(1),dVec(2),'g','LineWidth',2);
        quiver(m(1),m(2),toMouthVec(1),toMouthVec(2),'m','LineWidth',2);
        quiver(p0(1),p0(2),v1(1),v1(2),0,'r','LineWidth',2);
        quiver(p1(1),p1(2),v2(1),v2(2),0,'r','LineWidth',2);
        dot(v1,v2)
        %     showCoords(curLandmarks);
        
        dpc
    end
end
%%
direction_vecs = cat(1,direction_vecs{:});
to_mouth_vecs = cat(1,to_mouth_vecs{:});
to_mouth_vecs = normalize_vec(to_mouth_vecs,2);
direction_vecs = normalize_vec(direction_vecs,2);

function geom_descs = getGeomFeats(curLandmarks,mouthMask,regions,R)%,px,py)
outer_mouth_poly = curLandmarks(49:60,:);
inner_mouth_poly = curLandmarks(61:68,:);
% if polyarea2(inner_mouth_poly)==0        
% end
outer_mouth = poly2mask2(outer_mouth_poly,size2(mouthMask));
inner_mouth = poly2mask2(inner_mouth_poly,size2(mouthMask));

if nnz(outer_mouth) == 0
    outer_mouth = mouthMask;
end
if nnz(inner_mouth)==0
    inner_mouth = imerode(outer_mouth,ones(3));
    if nnz(inner_mouth)==0
        inner_mouth = zeros(size2(mouthMask));
        inner_mouth(sub2ind2(size(mouthMask),fliplr(round(mean(region2Pts(outer_mouth)))))) = 1;
    end
end
assert(nnz(inner_mouth)>0)
outer_mouth = outer_mouth & ~inner_mouth;
[outer_int,a1,a_out] = regionsInt(regions,{outer_mouth});
[inner_int,a1,a_in] = regionsInt(regions,{inner_mouth});
outer_int_1 = outer_int./a1(:);
outer_int_2 = outer_int/(a_out+eps);
inner_int_1 = inner_int./a1(:);
inner_int_2 = inner_int/(a_in+eps);
mouth_rad = round((nnz(outer_mouth)/pi)^.5);
rads = round([mouth_rad*[.5 1 2] size(mouthMask,1)/2]);
rads = max(rads,2);
for r = 2:length(rads)-1
    if rads(r)<=rads(r-1)+1
        rads(r) = rads(r)+2;
    end
end
%     rads = size(mouthMask,1)/2;
logPolarMask = getLogPolarMask(rads,15,2);
z = prod(size(mouthMask));
logPolarDescs = cellfun2(@(x) getLogPolarShape(x,[],[],logPolarMask)/z, regions);
logPolarDescs = cat(2,logPolarDescs{:});
assert(size(logPolarDescs,1)==46)
% check distance of inner,outer points to mouth masks...
d = bwdist_sign(mouthMask);
%R1 = round(size(mouthMask,1)*R(:,[2 1 4 3]));

minmaxDists = zeros(length(regions),2);
for iRegion = 1:length(regions)
    curDists = d(regions{iRegion});
    curMin = min(curDists);
    curMax = max(curDists);
    minmaxDists(iRegion,:) = [curMin curMax];
end
minmaxDists = minmaxDists/size(mouthMask,1);
geom_descs = [logPolarDescs;outer_int_1';outer_int_2';inner_int_1';inner_int_2';minmaxDists'];%;px';py'];