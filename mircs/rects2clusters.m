function clusters = rects2clusters(conf,rects,pos_ids,inds,toShow,minOvp,crop_out,sameCluster)

if (~iscell(rects))
    rects_ = {};
    for k = 1:size(rects,1)
        r = rects(k,:);
        r(3) = r(3)-r(1);
        r(4) = r(4)-r(2);
        rects_{k} = r;
    end
    rects = rects_;
end

if (nargin < 5)
    toShow = 0;
end
if (nargin < 6)
    minOvp = .2;
end
if (nargin < 7)
    crop_out = 0;
end

if (nargin < 8)
    sameCluster = false;
end
clust_count = 0;
conf.detection.params.detect_levels_per_octave = 8; % was 4
if (isempty(inds))
    inds = 1:(length(pos_ids));
end
for k = 1:length(rects)
    k
    currentID = pos_ids{k};
    I = getImage(conf,currentID);
    r = rects{k};
    r(3) = r(1)+r(3);
    r(4) = r(4)+r(2);
    
    if (size(r,2)>4 && r(7))
        disp('flipped')
        I = flip_image(I);
        r = flip_box(r,size(I));
    end
    
    %     r_inflate = inflatebbox(r,.8);
    if (crop_out)
        r_inflate = inflatebbox(r,1.5);
        I = cropper(I,r_inflate);
        r([1 3]) = r([1 3])-r_inflate(1);
        r([2 4]) = r([2 4])-r_inflate(2);
    end
    
    
%         [model] = conf.detection.params.init_params.init_function(I, r,...
%             conf.detection.params.init_params);
    
    [X,uus,vvs,scales,t] = allFeatures(conf,I);
    [ bbs ] = uv2boxes( conf,uus,vvs,scales,t );
    % find bb with highest intersection to current rect...
    ovp = boxesOverlap(bbs,r);
    [m,im] = max(ovp);
    
    if (m < minOvp)
        continue;
    end
    
    clust_count = clust_count+1;
    % [ overlaps ] = boxesOverlap( r,bbs);
    bb= bbs(im,:);
    bb(:,11) = inds(k);
    %bb(:,11) = pos_sel(k);
    %     Xs{k} = X(:,im);
    %     bbs_{k} = bb;
    clusters(clust_count) =  makeClusters(X(:,im),bb);
    
    if (toShow)
        figure(1); clf;
        imshow(I);
        hold on;
        plotBoxes2(r([2 1 4 3]),'lineWidth',2);
        plotBoxes2(bbs(im,[2 1 4 3]),'g','lineWidth',2);
        pause;
    end
    %pause;
end

if (sameCluster)
    clusters = makeCluster(cat(2,clusters.cluster_samples),cat(1,clusters.cluster_locs));
end

