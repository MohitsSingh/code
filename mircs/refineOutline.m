function res = refineOutline(conf,imageID,bb,outline,debug_)


conf.get_full_image = true; % work in full image coordinates for consistency
[I_orig,xmin,xmax,ymin,ymax] = getImage(conf,imageID);
if (min(dsize(I_orig,1:2)) < 30)
    res = [];
    return;
end
% bb = round(bb);
inflateFactor = 1.3;
bbs_new = inflatebbox(bb,inflateFactor*[1 1],'both',false);
bbs_new = clip_to_image(bbs_new,I_orig);
%outline = outline - repmat(bbs_new(1:2)-bb(1:2),size(outline,1),1);
outline = bsxfun(@plus,outline,bb(1:2)+[xmin ymin]);
%     bbs_new = round((round(inflatebbox(bb,[1 1],'both',false),I));
bbs_new = bbs_new + [xmin ymin xmin ymin];
bbs_new = round(bbs_new);
I = cropper(I_orig,bbs_new);
ucmFile = fullfile(conf.gpbDir,strrep(imageID,'.jpg','_ucm.mat'));
load(ucmFile); % ucm
% work in global image coordinates.
ucm = cropper(ucm,bbs_new);
regions_sub = combine_regions_new(ucm,.1);

% M = regionAdjacencyGraph(regions_sub);
% groups = enumerateGroups(M,3);
% generate the regions...
pp = poly2mask(outline(:,1)-bbs_new(1),outline(:,2)-bbs_new(2),size(I,1),size(I,2));
bestRegion = approximateRegion(pp,regions_sub,3);
% bestOVP = 0;
% bestRegion = regions_sub{1};
% for k = 1:length(groups)
%     curGroup = groups{k};
%     ii = k;
%     for kk = 1:size(curGroup,1)
%         curRegion = cat(3,regions_sub{curGroup(kk,:)});
%         curRegion = max(curRegion,[],3);
%         intersection_ = nnz(pp & curRegion);
%         union_ = nnz(pp | curRegion);
%         if (union_ > 0)
%             ovp = intersection_/union_;
%         else
%             ovp = 0;
%         end
%         %ovp = boxRegionOverlap(pp,curRegion);
%         if (ovp > bestOVP)
%             bestOVP = ovp;
%             bestRegion = curRegion;
%         end
%     end
% end

% regions_sub = newRegions;


% pp_test = imresize(pp,[80 NaN],'nearest');
% ovp = boxRegionOverlap(pp_test,newRegions);
% [p,ip] = sort(ovp,'descend');
% ij = regionIndex{ip(1)};
% newRegion = max(cat(3,regions_sub{curGroup(kk,:)}),[],3);
if (debug_)
    clf; subplot(1,3,1); imagesc(I); axis image; hold on;
    displayRegions(I,{bestRegion},bestOVP,-1,1);
    plot(outline(:,1)-bbs_new(1),outline(:,2)-bbs_new(2));
    pause;
end

bestRegion = imclose(bestRegion,ones(3));
u = false(size(I_orig));
% bb_t = clip_to_image(bbs_new,u);
u(bbs_new(2):bbs_new(4),bbs_new(1):bbs_new(3)) = bestRegion;
res = u(ymin:ymax,xmin:xmax);
