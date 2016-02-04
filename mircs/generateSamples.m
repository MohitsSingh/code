function [samples,sample_stats] = generateSamples(conf,fra_db,isPos,params,sample_stats)
if nargin < 5
    sample_stats = [];
end

dlib_landmark_split;
samples = struct('label',{},'roi_props',{},'roi',{},'img',{},'masks',{});
nSamples = 0;
tic_id = ticStatus('generating samples...',.5,.5);
for t = 1:length(fra_db)
    %     profile on
    % 1. get facial landmarks
    % 2. find outline of mouth region and edge of face (a proxy for pose)
    % 3. extract candidate regions, for now parameteric in nature
    % extract features: (1) geometric : angle, center, width (2) appearance
    % find negative samples, from non-class images.
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    % 1. facial landmarks (e.g, around mouth);
    [I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData);
%     mouthMask = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib);
%     labels = [];
    if isPos
        gt_graph = get_gt_graph(imgData,params.nodes,params,I);
        % 2. remove "trivial" regions
        if ~isfield(gt_graph{2},'roiMask')
            continue
        end
        m = gt_graph{2}.roiMask;
        if isempty(m) || nnz(m)==0
            continue
        end
        roiMask = cropper(m,mouthBox);
        if (nnz(roiMask)==0)
            continue
        end        
        % find angle, location of rectangle
        [r,fit_index] = fitRectangle(roiMask);
        pos_roi = directionalROI_rect_lite(r);
        roi_mask = poly2mask2(pos_roi,size2(I_sub));
        nSamples = nSamples+1;
        samples(nSamples).label = 1;
        samples(nSamples).roi_props = r/size(I_sub,1);
        samples(nSamples).roi = {pos_roi};
        samples(nSamples).img = I_sub;
        samples(nSamples).masks = {roi_mask};
    else
        assert(~isempty(sample_stats));       
        nToSample = 10;
        Y = random(sample_stats,nToSample);
        masks = {};
        rois = {};
        for tt = 1:nToSample
            %         clf; imagesc2(I_sub);
            %         plotPolygons(directionalROI_rect_lite(Y(t,:)*size(I_sub,1)));
            %         dpc
            rois{tt} = directionalROI_rect_lite(Y(tt,:)*size(I_sub,1));
            masks{tt} = poly2mask2(rois{tt},size2(I_sub));
        end
        
        nSamples = nSamples+1;
        samples(nSamples).label = -1;
        samples(nSamples).roi_props = Y;
        samples(nSamples).roi = rois;
        samples(nSamples).img = I_sub;
        samples(nSamples).masks = masks;
        samples(nSamples).imgIndex = imgData.imgIndex;
        %
    end
    
    tocStatus(tic_id,t/length(fra_db));
    
end

if nargout == 2 % compute samples stats.
    R = cat(1,samples.roi_props);
    n_gaussian_components = 2;
    sample_stats = fitgmdist(R,n_gaussian_components);%,'CovType','Diagonal');

end


