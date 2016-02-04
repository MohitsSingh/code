function graphData = constructGraph(conf,currentImage,prob_image,superPixMap)

rprops = regionprops(superPixMap,prob_image,'PixelIdxList','MeanIntensity');
% qq = zeros(size(prob_image));
% for k = 1:length(rprops)
%     qq(rprops(k).PixelIdxList) = rprops(k).MeanIntensity;
% end
% imagesc(qq);
prob_unary = [rprops.MeanIntensity];
ppp = prob_unary;
unary_ = -log([(1-ppp)+eps;ppp+eps]);

% build graph
props = {};
for k = 1:3
    props{k} = regionprops(superPixMap,currentImage(:,:,k),'MeanIntensity');
end

colors = [[props{1}.MeanIntensity];[props{2}.MeanIntensity];[props{3}.MeanIntensity]]+eps;
C = squeeze(vl_xyz2luv(vl_rgb2xyz(reshape(colors', [size(colors,2), 1, size(colors,1)]))))';
colors = C;
nsegs = size(colors,2);
D_ = vl_alldist2(colors,colors);
Ds = sum(D_(:));
Ds = Ds / ((nsegs*nsegs - nsegs));
% beta = 1/(2*Ds);
clear D_;
nSegs = length(unique(superPixMap));
% segs = seg_adj(superPixMap);
pairwise = zeros(nsegs, nsegs);
pairwises{2} = sparse(boundarylen(double(superPixMap),nSegs));

segs = [];
for k = 1:nSegs
    segs(k).adj = find(pairwises{2}(k,:));
end

for i = 1:nsegs
    for n = 1:length(segs(i).adj)
        j = segs(i).adj(n);
        pairwise(i,j) = 1/(1+norm(colors(:,i) - colors(:,j)));        
    end
end
pairwises{1} = sparse(pairwise);
pairwise_ = pairwises{1}.*pairwises{2};

graphData.unary = unary_;
graphData.pairwise = pairwise_;