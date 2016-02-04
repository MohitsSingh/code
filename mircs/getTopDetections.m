function [clusters] = getTopDetections(conf,detections,clusters,varargin)
%uniqueImages,useLocation,nDets
% find the top k detections for each cluster...

ip = inputParser;

checkX = @(x)((isscalar(x) || ismatrix(x) || iscell(x)));

ip.addParamValue('useLocation',0,checkX);
ip.addParamValue('perImageMasks',0,checkX);

ip.addParamValue('sals',{},@iscell);
ip.addParamValue('nDets',conf.clustering.top_k,@isnumeric);
ip.addParamValue('uniqueImages',true,@(x) islogical(x) || (isscalar(x) &&...
    any(intersect(x,[0 1]))));

ip.parse(varargin{:});
useLocation = ip.Results.useLocation;
top_k = ip.Results.nDets;
uniqueImages = ip.Results.uniqueImages;
sals = ip.Results.sals;

valids = find([clusters.isvalid]);
bbs = {};
xs = {};
for iDet = 1:length(detections) % iterate over all images
    for kk = 1:length(detections{iDet}.bbs)
        if (isempty(detections{iDet}.bbs{kk}))
            detections{iDet}.bbs{kk} = [];
        end
    end
    bbs{iDet} = detections{iDet}.bbs;
    
    xs{iDet} = detections{iDet}.xs;
    if (isnumeric(ip.Results.perImageMasks) && ~ip.Results.perImageMasks)
        continue;
    end
    
    
    curMask = ip.Results.perImageMasks{iDet};
    for k = 1:length(bbs{iDet})
        r = bbs{iDet}{k};
        r_c = round(boxCenters(r));
        vals = curMask(sub2ind(size(curMask),r_c(:,2),r_c(:,1)));
        r(:,conf.consts.SCORE) = vals.*(0.5*(2 + r(:,conf.consts.SCORE)));
        bbs{iDet}{k} = r;
    end
end



bbs = cat(2,bbs{:});
xs = cat(2,xs{:});

for iModel = 1:length(valids)
    %     iModel
    cluster_id = valids(iModel);
    
    lengths = zeros(1,length(detections));
    for iDet = 1:length(detections)
        % there cannot be detections from an invalid cluster. so access
        % bbs{iModel,...} instead of bbs{cluster_id}
        lengths(iDet) = size(bbs{iModel,iDet},1); %TODO!!! bbs is concatenated from valids only?
        % check it...
    end
    totalLength = sum(lengths);
    inds = zeros(1,totalLength);
    c = 0;
    for k = 1:length(lengths)
        inds(c+1:c+lengths(k)) = k;
        c = c+lengths(k);
    end
    % there cannot be detections from an invalid cluster. so access
    % bbs{iModel,...} instead of bbs{cluster_id}
    model_bboxes = cat(1,bbs{iModel,:});
    % hold on; plotBoxes2(model_bboxes(:,[2 1 4 3]));
    z = false;
    if (iscell(useLocation) || numel(useLocation)>1 || useLocation)
        bbox_centers = boxCenters(model_bboxes);
        z = true;
        %         if (isscalar(useLocation))
        %             curCenter = clusters(cluster_id).refLocation;
        %             dists = l2(curCenter,bbox_centers);
        %             validDists = exp(-dists(:)/useLocation);
        %         else
        
        if (iscell(useLocation))
            %             figure,imshow(useLocation{iModel});
            %             hold on;
            %             for qq = 1:size(bbox_centers,1)
            %                 plot(round(bbox_centers(qq,1)),round(bbox_centers(qq,2)),'r+');
            %                 pause
            %             end
            %             hold on;
            %             plotBoxes2(model_bboxes(:,[2 1 4 3]),'g');
            % %
            bbox_centers(:,1) = max(bbox_centers(:,1),1);
            bbox_centers(:,2) = max(bbox_centers(:,2),1);
            bbox_centers(:,1) = min(bbox_centers(:,1),size(useLocation{iModel},2));
            bbox_centers(:,2) = min(bbox_centers(:,2),size(useLocation{iModel},1));
            validDists = useLocation{iModel}(sub2ind(size(useLocation{iModel}),round(bbox_centers(:,2)),round(bbox_centers(:,1))));
            
        elseif(ismatrix(useLocation))
            validDists = useLocation(sub2ind(size(useLocation),...
                round(bbox_centers(:,2)),round(bbox_centers(:,1))));
        else
            error('use location must be scalar, matrix or cell');
        end
        %     end
        %validDists = dists < 3;
    else
        validDists = ones(size(model_bboxes,1),1);
    end
    
    
    % remove models for which there were too few detections.
    nScores = size(model_bboxes,1);
    if (nScores < conf.clustering.min_cluster_size)
        clusters(cluster_id).isvalid = false;
        clusters(cluster_id).cluster_locs = [];
        clusters(cluster_id).cluster_samples = [];
        continue;
    end
    % there cannot be detections from an invalid cluster. so access
    % xs{iModel,...} instead of xs{cluster_id}
    model_xs =  cat(2,xs{iModel,:});
    
    
    if (z)
        model_bboxes(:,conf.consts.SCORE) = .5*(2 + model_bboxes(:,conf.consts.SCORE)).*validDists;
    end
    
    if (~isempty(sals))
        box_centers = round(boxCenters(model_bboxes));
        for iSal = 1:length(sals)
            sel = inds==iSal;
            bc_xy = box_centers(sel,:);
            v = im2double(sals{k}(sub2ind(size(sals{k}),bc_xy(:,2),bc_xy(:,1))));
            %v = double(v>.2);
            ff = .5*(1.5 + model_bboxes(sel,conf.consts.SCORE));
            model_bboxes(sel,conf.consts.SCORE) = v.*ff;
        end
    end
    
    
    det_scores = model_bboxes(:,conf.consts.SCORE);
    if (isfield(clusters,'xy'))
        mean_xy = mean(clusters(cluster_id).xy,2)';
        mean_det_xy = [mean(model_bboxes(:,[1 3]),2),mean(model_bboxes(:,[2 4]),2)];
        pos_distance = sum((bsxfun(@minus,mean_det_xy,mean_xy)).^2,2).^.5/conf.clustering.secondary.img_size;
        det_scores = det_scores+0*(1./(1+5*pos_distance));
    end
    
    
    
    [det_scores,iScores] = sort(det_scores,'descend');
    % score according to distance from center.
    nScores = length(det_scores);
    locs = model_bboxes(iScores,:);
    locs(:,11) = inds(iScores);
    
    if (uniqueImages)
        [a,b,c] = unique(locs(:,11),'first');
        locs = locs(b,:);
        nScores = size(locs,1);
        [~,iScores_u] = sort(locs(:,12),'descend');
        validDists = validDists(b(iScores_u));
        iScores_u = iScores_u(1:min(top_k,nScores));
        validDists = validDists(1:min(top_k,nScores));
        locs = locs(iScores_u,:);
    end
    %
    %     if (z) % score the locations only after the best unique location has been detected.
    %         locs(:,conf.consts.SCORE) = .5*(2 + locs(:,conf.consts.SCORE)).*validDists;
    %         %     model_bboxes(:,conf.consts.SCORE) = (model_bboxes(:,conf.consts.SCORE));
    %     end
    %
    clusters(cluster_id).cluster_locs = locs((1:min(top_k,nScores)),:);
    if (~isempty(model_xs))
        m =model_xs(:,iScores);
        if (iscell(m))
            m = cat(2,m{:});
        end
        
        if (uniqueImages)
            m = m(:,b);
            m = m(:,iScores_u);
        end
        
        clusters(cluster_id).cluster_samples = m(:,1:min(top_k,nScores));
    else
        % %         fprintf(2,'cluster %03.0f has no saved samples\n',iModel);
    end
    
    % finally, sort one more time...
    [s_,is_] = sort(clusters(cluster_id).cluster_locs(:,12),'descend');
    clusters(cluster_id).cluster_locs = clusters(cluster_id).cluster_locs(is_,:);
    if (size(clusters(cluster_id).cluster_samples,2) == length(is_))
        clusters(cluster_id).cluster_samples = clusters(cluster_id).cluster_samples(:,is_);
    end
end
