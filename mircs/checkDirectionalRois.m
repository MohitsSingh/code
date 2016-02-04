function regionGroups = checkDirectionalRois(segLabels,startPts2,im,origBoundaries,UCM,regions,bbox)
% checkDirectionalRois examine segment sequences in different directions.
dT = 15;
%theta = (0:dT:360-dT)';
theta = (-50:dT:50)';
vec = [sind(theta) cosd(theta)];
% plot(vec(:,1),vec(:,2),'r-');
start_ = repmat(startPts2,length(theta),1);
end_ = start_+max(size(im))*vec;

pts = {};
rois = {};
if (nargin < 6)    
    regions = {};
    for k = 1:length(unique(segLabels))
        regions{k} = segLabels==k;
    end
else
    regions = fillRegionGaps(regions,true);
end
regionGroups = struct('regionSubset',{},'theta',{});
debug_ = false;

for k = 1:size(start_,1)
    k
    if (nargin == 7)
        xmin = bbox(1);
        xmax = bbox(3);
        ymin = bbox(2);
        ymax = bbox(4);
        x = [start_(k,1) xmin xmin xmax xmax];
        y = [start_(k,2) ymin ymax ymax ymin];
        thinRoi = roipoly(im,x,y);
        wideRoi = thinRoi;
        
        roi = directionalROI(im,start_(k,:),vec(k,:)',10);
        ints_line = cellfun(@(x) sum(x(:) & thinRoi(:)),regions);
        ints = cellfun(@(x) sum(x(:) & wideRoi(:)),regions);
        areas = cellfun(@(x) sum(x(:)), regions);
        
        coverage = (ints./areas);
        goodRegions = coverage > .7 & ints_line > 0
        if (~any(goodRegions))
            continue;
        end
        regionUnion = cat(3,regions{goodRegions});
        regionUnion = max(regionUnion,[],3);
        %     ints = cellfun(@(x) sum(x(:) & Z_sel),regions);
        %     areas = cellfun(@(x) sum(x(:)), regions);
        %     uns = cellfun(@(x) sum(x(:) | Z),regions);
        [yy1,xx1] = find(bwperim(thinRoi));
        [yy2,xx2] = find(bwperim(wideRoi));
        %     curLabel = ismember(segLabels,unique(segLabels(pts_sub)));
        displayRegions(bsxfun(@times,im,origBoundaries),{regionUnion},1,-1);
        %         displayRegions(im,{regionUnion},1,-1);
        hold on;
        plot(xx1,yy1,'g.');
        plot(xx2,yy2,'m.');
        %         pause;
        break;
    end
    thinRoi = directionalROI(im,start_(k,:),vec(k,:)',10);
    wideRoi = directionalROI(im,start_(k,:),vec(k,:)',30);
    ints_line = cellfun(@(x) sum(x(:) & thinRoi(:)),regions);
    ints = cellfun(@(x) sum(x(:) & wideRoi(:)),regions);
    areas = cellfun(@(x) sum(x(:)), regions);
    
    coverage = (ints./areas);
    goodRegions = coverage > .7 & ints_line > 0;
    
    regionGroups(k).theta = theta(k);
    regionGroups(k).regionSubset = find(goodRegions);
    
    if (~any(goodRegions))
        continue;
    end
    
    %     ints = cellfun(@(x) sum(x(:) & Z_sel),regions);
    %     areas = cellfun(@(x) sum(x(:)), regions);
    %     uns = cellfun(@(x) sum(x(:) | Z),regions);
    
    if (debug_)
        regionUnion = cat(3,regions{goodRegions});
        regionUnion = max(regionUnion,[],3);
        [yy1,xx1] = find(bwperim(thinRoi));
        
        [yy2,xx2] = find(bwperim(wideRoi));
        %     curLabel = ismember(segLabels,unique(segLabels(pts_sub)));
        %displayRegions(bsxfun(@times,im,origBoundaries),{regionUnion},1,-1);
        displayRegions(im,{regionUnion},1,-1);
        hold on;
        % %     plot(xx1,yy1,'g.');
        % %     plot(xx2,yy2,'m.');
        pause;
    end
    %     end
end



end