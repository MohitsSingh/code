function segData = getSegData(conf,imgData,debug_)
if (nargin < 3)
    debug_ = false;
end
mm = 2; nn = 3;
segData = struct('props',{},'pairScores',{},'lineScores',{},'totalScores',{});
segData(1).props = [];
if (imgData.faceScore < -1)
    return;
end
[M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,imgData,conf.straw.extent,true);
M_orig = M;

% [regions,regionOvp,G] = getRegions(conf,imgData.imageID);
% regions1 = multiCrop(conf,regions,round(face_box));
% regions1=regions1(cellfun(@(x) nnz(x)>5,regions1));
% regions2 = removeDuplicateRegions(regions1);
% displayRegions(M,regions1);



sz = conf.straw.dim;
sz = [sz sz];
s1 = size(M,1);
M = imresize(M,sz,'bicubic');
[mouth_poly] = fix_mouth_poly(mouth_poly);
mouth_poly_2 = bsxfun(@minus,mouth_poly,face_box(1:2));
mouth_poly_2 = mouth_poly_2*conf.straw.dim/s1;
face_poly_2 = bsxfun(@minus,face_poly,face_box(1:2));
face_poly_2 = face_poly_2*conf.straw.dim/s1;
face_mask_2 = poly2mask2(face_poly_2,size2(M));

[ucm,gPb_thin] = loadUCM(conf,imgData.imageID); %#ok<*STOUT>

mouth_box_2 = pts2Box(mouth_poly_2);
mouth_mask = poly2mask2(mouth_poly_2,size2(M));
mouth_dist = bwdist(mouth_mask);
M = clip_to_bounds(M);
gPb_thin = cropper(gPb_thin,round(face_box));

% ucm = imResample(ucm,size(M),'bilinear');

[lineness,symmetry,mag,orientation] = getLineProps(M);

edgeMap = edge(rgb2gray(M),'canny');
[edgelist edgeim] = edgelink(edgeMap, []);
[seglist,inds] = lineseg(edgelist,2);

M1 = imfilter(rgb2gray(M),fspecial('gauss',7,3));
edgeMap = edge(M1,'canny');
[edgelist1] = edgelink(edgeMap, []);
[seglist1,inds] = lineseg(edgelist1,2);
seglist  =[seglist seglist1];

M1 = imfilter(rgb2gray(M),fspecial('gauss',9,5));
edgeMap = edge(M1,'canny');
[edgelist2] = edgelink(edgeMap, []);
[seglist2,inds] = lineseg(edgelist2,2);
seglist  =[seglist seglist2];

edgeMap = gPb_thin>.2;
if (nnz(edgeMap) > 5)
    [edgelist edgeim] = edgelink(edgeMap, []);
    [seglist3,inds] = lineseg(edgelist,2);
    seglist3 = cellfun2(@(x) round(x*conf.straw.dim/s1),seglist3);
    seglist  =[seglist seglist3];
end
% find for each segment it's original edge, to later check which segments
% belong to the same connected component.
% % % R = bwlabel(edgeMap);

% % % original_edge = {};
% % % for k = 1:numel(seglist)
% % %     r = R(sub2ind2(size(R),seglist{k}(1,:)));
% % %     original_edge{k} = r*ones(size(seglist{k},1)-1,1);
% % %     %original_edge{k} = k*ones(size(seglist{k},1)-1,1);
% % % end
% % % original_edge = cat(1,original_edge{:});
if (debug_)
    clf;
    [I,I_rect] = getImage(conf,imgData);
    
    subplot(mm,nn,1);imagesc2(I);
    plotBoxes(face_box,'g--');
    plotBoxes(mouth_box,'m');
    plotPolygons(face_poly,'r--');
    plotPolygons(mouth_poly);
    showCoords(mouth_poly,'Color','g');
    subplot(mm,nn,2); imagesc2(M); colormap gray;
    subplot(mm,nn,4); imagesc2(mag);
    subplot(mm,nn,3);
    imagesc2(edgeMap);colormap gray;
    
end
% sel
segs = seglist2segs(seglist);
segs = segs(:,[2 1 4 3]); % make order x,y,x,y
segs = fixSegs(segs); % make sure that the upper y point is first.

segs = removeDuplicateSegs(segs);

startPts = segs(:,1:2);
% define region of interest:

tt = conf.straw.dim;
r = round(tt/10);

sel_x = ismember(startPts(:,1),r:tt-r);
sel_y = ismember(startPts(:,2),r*2:8*r);
sel_ = sel_x & sel_y;
startPts = startPts(sel_,:);
f_sel = find(~sel_);
if (nnz(sel_)<2) % takes two to tango
    return;
end
segs = segs(sel_,:);
% original_edge = original_edge(sel_);

[Z,allPts]= paintLines(zeros(size2(M)),segs);

% calculate face perimeter to remove edges with too many pixels in this
% region.
face_perim = bwperim(face_mask_2);
face_perim = imdilate(face_perim,ones(3));
face_perim = addBorder(face_perim,3,0);

min_mouth_dists = zeros(size(allPts));
max_mouth_dists = zeros(size(allPts));
area_in_perim = zeros(size(allPts));
n_in_mouth = zeros(size(allPts));
% z0 = zeros(size2(M));
y_nears = zeros(size(allPts));
y_fars = zeros(size(allPts));
for k = 1:length(min_mouth_dists)
    pt = sub2ind2(size2(M),allPts{k}(:,[2 1]));
    area_in_perim(k) = sum(face_perim(pt))/length(pt);
    curDists = mouth_dist(pt);
    [min_,imin] = min(curDists);
    [max_,imax] = max(curDists);
    y_nears(k) = allPts{k}(imin,2);
    y_fars(k) = allPts{k}(imax,2);
    min_mouth_dists(k) = min_;
    max_mouth_dists(k) = max_;
    n_in_mouth(k) = mean(mouth_dist(pt)==0);
    %     z0(pt) = k;
end

vecs = segs2vecs(segs);
if (debug_)
    quiver(segs(:,1),segs(:,2),vecs(:,1),vecs(:,2),0,'r','LineWidth',2);
    subplot(mm,nn,5);
    imagesc2(Z);    
end
% 1. segs with strong responses...
...segs(:,[2 1 4 3]));
    % compute the different attributes for each segment...
Z = imdilate(Z,ones(3));
rprops = regionprops(Z,'PixelIdxList');
props = zeros(size(segs,1),8);
d = atand(vecs(:,2)./vecs(:,1));

ys = segs(:,[2 4]); xs = segs(:,[1 3]);
top_y = min(ys,[],2);
bottom_y = max(ys,[],2);
props(:,1) = d;
for r = 1:length(rprops) %TODO - this is not optimal since some lines are drawn over other and so
    % the properties are not calculated correctly.
    p = rprops(r).PixelIdxList;
    props(r,2) = mean(symmetry(p));
    props(r,3) = mean(lineness(p));
    props(r,4) = size(allPts{r},1); % area, not length. correct this.
    props(r,7) = mean(mag(p));
    props(r,8) = top_y(r);
    props(r,9) = min_mouth_dists(r);
    props(r,10) = max_mouth_dists(r);
    props(r,11) = bottom_y(r);
    props(r,12) = mean(face_mask_2(sub2ind2(size2(M),fliplr(allPts{r})))); % pct. straw in face mask;
    props(r,13) = area_in_perim(r);
end
props(:,5:6) = startPts;

% get width and left-right corners of mouth.

mouth_data = get_mouth_data(mouth_poly_2);

[pairScores,lineScores,pairExplanations,lineExplanations,pairPolys] = getStrawScores(conf,segs,props,mouth_box_2,M,mouth_data,imgData,debug_);
gpb_reverse_evergy = zeros(size(pairScores));

for i1 = 1:size(pairPolys,1)
    for i2 = 1:size(pairPolys,2)
        curPoly = pairPolys{i1,i2,1}/conf.straw.dim*s1;
        curPoly_reverse = pairPolys{i1,i2,2}/conf.straw.dim*s1;
        if (any(curPoly))
            
            bw = poly2mask2(curPoly_reverse,size2(gPb_thin));
            gpb_reverse_evergy(i1,i2) = mean(gPb_thin(bw(:)));
%             i1 = 77
%             i2 = 55           
%             clf;imagesc2(gPb_thin); 
%             plotPolygons(curPoly,'g-');
%             plotPolygons(curPoly_reverse,'y-');
%             title(num2str(gpb_reverse_evergy(i1,i2) ));
%             pause
        end
    end
end
        

% lineScores = lineScores + exp(-min_mouth_dists/(conf.straw.dim/10))';


really_fars = min_mouth_dists > conf.straw.dim/5;
[v1,v2] = meshgrid(really_fars,really_fars);
really_fars = v1 | v2;
pairScores(really_fars) = -100;
pairExplanations = appendString(pairExplanations,really_fars,[],'at least one very far from mouth',debug_);

really_fars_2 = min_mouth_dists > conf.straw.dim/10;
[v1,v2] = meshgrid(really_fars_2,really_fars_2);
really_fars = v1 & v2;
pairScores(really_fars) = -100;
pairExplanations = appendString(pairExplanations,really_fars,[],'both far from mouth',debug_);


% have to start in face, and be somewhat contained in it.
in_face = props(:,12);
in_face = in_face > .1;
[v1,v2] = meshgrid(in_face,in_face);
%not_in_face = ~(~v1 | v2);
not_in_face = (~v1 | ~v2); % both must be in face (at least partially)
% fars = unique([ii;jj]);
pairScores(not_in_face) = -100;
pairExplanations = appendString(pairExplanations,not_in_face,[],'not in face',debug_);

% cannot have one totally outside face if other is totally inside
[v1,v2] = meshgrid( props(:,12));
half_in_face = (v1==0 & v2==1) | v1==1 & v2==0;
pairScores(half_in_face) = -100;
pairExplanations = appendString(pairExplanations,half_in_face,[],'only half in face',debug_);

lineScores(max_mouth_dists ==0) = -inf;
lineExplanations = appendString(lineExplanations,col(max_mouth_dists ==0),[],'ends in mouth',debug_);
lineScores(y_nears > y_fars) = -inf;
lineExplanations = appendString(lineExplanations,col(y_nears > y_fars),[],'y_nears > y_fars',debug_);

% lineScores(n_in_mouth>=.8) = -inf;
% % % % [i1,i2] = find(pairScores>0);
% % % % ii = sub2ind2(size(pairScores),[i1 i2]);
% % % % diffs = original_edge(i1)~=original_edge(i2);
% % % % t_p = false(size(pairScores));t_p(ii(diffs)) = true;

% t_p = ii(t(:,1)~=t(:,2));
% % % % pairScores(t_p) = -inf; %TODO
% % % % pairExplanations = appendString(pairExplanations,t_p,[],'not connected',debug_);
% lineScores(f_sel) = -inf;

[v1,v2] = meshgrid(lineScores,lineScores);
lambda = 1;

segData.pairScores = pairScores;
segData.lineScores = lineScores;
segData.props = props;
segData.segs = segs;
segData.pairExplanations = pairExplanations;
segData.lineExplanations = lineExplanations;
totalScores = pairScores + lambda*(v1+v2);
totalScores(eye(size(totalScores))>0) = -inf;
segData.totalScores = totalScores;

%%%%%%%
% now diversify the list by removing near-duplicates.
[a1,a2] = meshgrid(1:size(totalScores,1));
totalScores(a1 <= a2) =-inf;
[ii,jj,vv] = find(totalScores);

%%



%%

inds = find(totalScores(:));
ii = ii(vv>0);jj=jj(vv>0);
inds = inds(vv>0);
vv = vv(vv>0);
ij = [ii jj];

if (isempty(vv))
    return;
end

% create polygons from candidates to find pixel-wise overlap
polygons = cell(length(vv),1);
for qq = 1:length(vv)
    ii = ij(qq,1); jj = ij(qq,2);
    xy = reshape([segs(ii,:)';segs(jj,:)'],2,[])';
    xy = xy(convhull(xy),:);
    polygons{qq} = xy;
end
% find the intersections of all polygons
allMasks=cellfun2(@(x) poly2mask2(x,size2(M)),polygons);
[ovp] = regionsOverlap(allMasks,allMasks);
T_ovp = .5;

%pair_subset = ij(subset,:);

subset = suppresRegions(ovp,T_ovp,row(vv));
subset = cat(2,subset{:});
seg_subset = unique(col(ij(subset,:)));



F = true(size(lineScores));F(seg_subset) = false;

%Z = -100;

% pairScores(F,F) = Z;
% totalScores(F) = Z;
% lineScores(F) = Z;
segData.pairScores(F,F) = pairScores(F,F)-1;
segData.lineScores(F) = lineScores(F)-1;
segData.gpb_reverse_evergy = gpb_reverse_evergy;
totalScores = pairScores + lambda*(v1+v2);
totalScores(eye(size(totalScores))>0) = -inf;
[dists1,dists2] = meshgrid(min_mouth_dists);
nearMouth = dists1 == 0 & dists2 == 0;

% if there is anyone touching the mouth, discard those not touching

isNearMouth = totalScores > 0 & nearMouth;
if (any(isNearMouth(:)))
    totalScores(~isNearMouth) = 0;
end

%totalScores(nearMouth) = totalScores(nearMouth)+max(0,max(totalScores(~nearMouth)));%TODO
segData.totalScores = totalScores;




%%%%%%%%%

f = j2m('/home/amirro/storage/occluders_s40_new4', imgData);
segData.wasOccluded = false;
segData.scoreBoost = 0;
L = load(f);
if (~isempty(L.rprops))
    % boost (or don't) score of totally-inside-face by examining occluders.
    [s,is] = max(totalScores(:));
    if (s < -10) % no point in doing this...
        return;
    end
    [ii,jj] = ind2sub(size(totalScores),is);
    [I,I_rect] = getImage(conf,imgData);
    disp(imgData.imageID);
    xy = reshape([segData.segs(ii,:)';segData.segs(jj,:)'],2,[])';
    xy = xy(convhull(xy),:);
    % shift the polygon back to the original image.
    xy = xy/(conf.straw.dim/s1);
    xy = bsxfun(@plus,xy,face_box(1:2));
    candidateMask = poly2mask2(xy,size2(I));
    % subplot(mm,nn,4),imagesc2(candidateMask);
    %check the intersection between this polygon and the major
    %occluders.
    seg_scores = get_putative_occluders(conf,imgData,L);
    occludingRegions = L.occludingRegions;
    candidateArea = nnz(candidateMask);
    portionsInFace = segData.props([ii jj],12);
    
    if (min(portionsInFace) == 1) % totally in face, check other occluders
        segData.wasOccluded = true;
        %     disp('totally inside face, checking occluders...')
        [ovp,ints,uns] = regionsOverlap(candidateMask,occludingRegions);
        score_boost = max(double(ints./candidateArea>0).*seg_scores);
        %     disp(['score boost: ' num2str(score_boost)]);
        segData.scoreBoost = score_boost;
    end
end
function mouth_data = get_mouth_data(mouth_poly)
[mouth_width,iWidth] = max(col(l2(mouth_poly,mouth_poly)));
mouth_width = mouth_width^.5;
[w1,w2] = ind2sub(size(mouth_poly,1)*[1 1],iWidth);
pt_1 = mouth_poly(w1,:); pt_2 = mouth_poly(w2,:);
if (pt_1(1) > pt_2(1))
    pt_1 = mouth_poly(w2,:);pt_2 = mouth_poly(w1,:);
end
mouth_data.mouth_width = mouth_width;
mouth_data.pt_1 = pt_1; mouth_data.pt_2 = pt_2;
mouth_data.mouth_poly = mouth_poly;

function segs = removeDuplicateSegs(segs)
segs = unique(segs,'rows');

