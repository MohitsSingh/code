function extractCandidateFeatures2(conf,imageSet,salData,k,debug_)
%Rs = extractCandidateFeatures3(conf,imageSet,salData,k)
if (nargin < 5)
    debug_ = false;
end
curFeats = struct('bbox',{},'bc',{},'horz_extent',{},'y_top',{},'ucmStrength',{},'isConvex',{},...
    'salStrength',{});

currentID = imageSet.imageIDs{k};
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
load(ucmFile); % ucm
ucm = ucm(ymin:ymax,xmin:xmax); %#ok<NODEF>
bbox = round(imageSet.faceBoxes(k,1:4));
bbox = clip_to_image(bbox,I);
subUCM = ucm(bbox(2):bbox(4),bbox(1):bbox(3));
if (isempty(subUCM))
    return;
end
I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
E = subUCM;


xy = faceLandmarks.xy;
xy_c = boxCenters(xy);

chull = convhull(xy_c);
% find the occluder!! :-)
c_poly = xy_c(chull,:);
c_poly = bsxfun(@minus,c_poly,bbox(1:2));
face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(E,1),size(E,2));

[ii,jj,vv] = find(subUCM);
if (length(ii) < 5)
    return;
end
% boundaries = bwbounadaries(face_mask);

E = E>.2;
if (~any(E(:)))
    return
end
E = bwmorph(E,'thin',Inf);
if (nnz(E) < 5)
    return
end

if (debug_)
end

regions_sub = combine_regions_new(subUCM,.1);
regions_sub = fillRegionGaps(regions_sub);
areas = cellfun(@nnz,regions_sub);
regions_sub(areas/numel(E)>.6) = [];

% perims = cellfun2(@bwperim,regions_sub);
boundaries = cellfun2(@bwboundaries,regions_sub);

% remove pixels on the edge the image.

% take only one of each.
boundaries = cellfun2(@(x) x{1},boundaries);
boundaries_new ={};
% % clf; subplot(1,2,1); imagesc(I_sub_color); axis image;
% % subplot(1,2,2);  imagesc(subUCM); axis image; pause;
for iB = 1:length(boundaries)
%     iB = 5
    cb = boundaries{iB};
%     iB
    onBorder = cb(:,1)==1 | cb(:,2)==1 | cb(:,2)==size(E,2) | cb(:,1)==size(E,1);
    
    % rearrange so that the first occurence of border (if any) is first, to
    % maintain continuity)
    
    cb_2 = {};
    
    
    isBorder = onBorder(1);
    curStart = 1;
    curEnd = 0;
    for k = 2:length(onBorder)
        if (isBorder)
            if (onBorder(k)) % still a border
                continue;
            else % new start, finish previous sequence
                isBorder = false;
                curStart = k;
            end
        else
            if (onBorder(k)) % end current sequence
                isBorder = true;
                curEnd = k-1;
                cb_2{end+1} = cb(curStart:curEnd,:);
            else  % still in sequence, do nothing
            end
        end
    end
    
    % ended with no border...
    if (~isBorder && curEnd < curStart)
        curEnd = length(onBorder);
        cb_2{end+1} = cb(curStart:curEnd,:);
    end
    
    
    % now check the endpoint of each sub-segment, to re-connect those which
    % were accidentally connected.
    if (length(cb_2)==1)
        cb_3 = cb_2;
    else
        cb_3 = {};
        u = false(size(cb_2));
        for i1 = 1:length(cb_2)
            i2 = i1+1;
            if (i1 == length(cb_2))
                i2 = 1;
                if (u(i1))
                    break;
                end
            end
            s1 = cb_2{i1}(1,:);
            e1 = cb_2{i1}(end,:);
            s2 = cb_2{i2}(1,:);
            e2 = cb_2{i2}(end,:);
            
            if max(abs(s1-s2),[],2)<=1 % <-- -->
                cb_3{end+1} = [cb_2{i1};flipud(cb_2{i2})];
                u([i1 i2]) = true;
            elseif (max(abs(e1-e2),[],2)<=1)
                cb_3{end+1} = [cb_2{i1};flipud(cb_2{i2})]; % --> <--
                u([i1 i2]) = true;
            elseif (max(abs(s1-e2),[],2)<=1)
                cb_3{end+1} = [cb_2{i1};cb_2{i2}]; % <-- <--
                u([i1 i2]) = true;
            elseif (max(abs(e1-s2),[],2)<=1) %--> -->
                cb_3{end+1} = [cb_2{i1};cb_2{i2}];
                u([i1 i2]) = true;
            end            
        end
        cb_3 = [cb_3,cb_2(~u)];
    end
    
    boundaries_new = [boundaries_new,cb_3];
%     
%     for k = 1:length(cb_3)
%         imagesc(E); axis image; hold on;
%         plot(cb(:,2),cb(:,1),'r.')
%         plot(cb_3{k}(:,2),cb_3{k}(:,1),'g.')
%         pause;
%     end
     

end

lines = lineseg(boundaries_new,2);
% lines = lineseg(cb_3,2);
%
% for k = 1:length(boundaries)
%     k
%     clf; imagesc(regions_sub{k}); axis image; hold on;
%     drawedgelist(lines(k),size(regions_sub{k}));
%     pause;
% end

[candidates,inds] = splitByDirection(lines);
inds = [inds{:}];
% make sure left point is first.
candidates = fixSegLists(candidates);
candidates = seglist2edgelist(candidates);
if (isempty(candidates))
    return
end
