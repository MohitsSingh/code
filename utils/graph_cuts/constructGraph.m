function graphData = constructGraph(currentImage,prob_image,superPixMap,killBorders,ucmMAP)

rprops = regionprops(superPixMap,prob_image,'PixelIdxList','MeanIntensity','Centroid');
seg_probImage = zeros(size(prob_image));
if (nargin < 4)
    killBorders = 0;
end
if (killBorders)
    borderImage = ones(size(prob_image));
    borderImage(killBorders+1:end-killBorders,killBorders+1:end-killBorders) = 0;
    borderPixels = find(borderImage);
end

for k = 1:length(rprops)
    curPixels = rprops(k).PixelIdxList;
    if (killBorders)
        if (~isempty(intersect(borderPixels,curPixels)))
            rprops(k).MeanIntensity = 0;
        end
    end    
end

% imagesc(qq);
prob_unary = [rprops.MeanIntensity];
% prob_unary = normalise(prob_unary);

for k = 1:length(rprops)
    seg_probImage(rprops(k).PixelIdxList) = prob_unary(k);
end

ppp = prob_unary;
ppp = normalise(ppp);
unary_ = -log([(1-ppp)+eps;ppp+eps]);

% build graph
props = {};
for k = 1:3
    props{k} = regionprops(superPixMap,currentImage(:,:,k),'MeanIntensity');
end

colors = [[props{1}.MeanIntensity];[props{2}.MeanIntensity];[props{3}.MeanIntensity]]+eps;
C = squeeze(vl_xyz2lab(vl_rgb2xyz(reshape(colors', [size(colors,2), 1, size(colors,1)]))))';
colors = C;
nsegs = length(rprops);
% D_ = vl_alldist2(colors,colors);
% Ds = sum(D_(:));
% Ds = Ds / ((nsegs*nsegs - nsegs));
% clear D_;
nSegs = length(unique(superPixMap));
pairwise = zeros(nsegs, nsegs);
pairwises{2} = sparse(boundarylen(double(superPixMap),nSegs));

segs = [];
for k = 1:nSegs
    segs(k).adj = find(pairwises{2}(k,:));
end
% pairwise3 = zeros(size(pairwise));
for i = 1:nsegs
    c1 = round(rprops(i).Centroid);
    for n = 1:length(segs(i).adj)
        j = segs(i).adj(n);
        c2 = round(rprops(j).Centroid);
        [rr,cc] = LineTwoPnts(c1(2),c1(1),c2(2),c2(1));
%         clf; imagesc(ucmMAP);hold on; plot(c1(1),c1(2),'g*');
%         plot(c2(1),c2(2),'g+');
%         plot(cc,rr,'m--');
%         pairwise3(i,j) = max(ucmMAP(sub2ind2(size(ucmMAP),[rr;cc]')));
        pairwise(i,j) = 1/(1+norm(colors(:,i) - colors(:,j)));
        
        
        
    end
end
% pairwises{3} = sparse(pairwise3);
pairwises{1} = sparse(pairwise);
%pairwise_ = pairwises{1}.*pairwises{2};

graphData.unary = unary_;
graphData.pairwise = pairwises;
graphData.seg_probImage = seg_probImage;