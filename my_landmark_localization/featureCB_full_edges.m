function psi = featureCB_full_edges(param, x, y)
% disp('feature');

nPts = param.nPts;
nEdges = param.nEdges;
cellSize = param.cellSize;
if (length(size(x))==2)
    x = cat(3,x,x,x);
end

u = y(:,1:2)*param.imgSize;
sub_rect = [u u];
sub_rect = inflatebbox(sub_rect,param.windowSize,'both',true);

% sub_rect = inflatebbox(param.imgSizerepmat(y(:,1:2),1,2),windowSize,'both',true);

subImgs = multiCrop2(x,round(sub_rect));
%%
X = getImageStackHOG(subImgs,param.windowSize,true,false,cellSize);
X = sparse(double(X));

psi = zeros(param.dimension,1);
psi(1:numel(X)) = X(:);
n = numel(X);


for iEdge = 1:nEdges
%     iu
    iPair = param.edges(iEdge,1);
    jPair = param.edges(iEdge,2);
    cur_offset = squeeze(y(jPair,1:2)-y(iPair,1:2));
    psi(n+1:n+5) = [cur_offset cur_offset.^2 1];
    n = n+5;
    %psi(nPts+iEdge)= pdf(param.pair_gaussian_models{iEdge},-cur_offset(1:2));    
end

psi = sparse(psi);

end