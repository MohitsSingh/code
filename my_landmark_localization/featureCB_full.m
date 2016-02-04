function psi = featureCB_full(param, x, y)
% disp('feature');

% nPts = param.nPts;
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
% X = sparse(double(X));
% X(end+1,:) = 1;
% if (param.use_location_prior)
%     
% end
psi = zeros(param.dimension,1);
psi(1:numel(X)) = X(:);
n = numel(X);
for iEdge = 1:nEdges
    %     iu
    iPair = param.edges(iEdge,1);
    jPair = param.edges(iEdge,2);
    cur_offset = squeeze(y(jPair,:)-y(iPair,:));
    %psi(nPts+iEdge)= pdf(param.pair_gaussian_models{iEdge},-cur_offset(1:2));
    psi(n+iEdge)= log_lh(param.pair_gaussian_models{iEdge},cur_offset(1:2));
end

psi = sparse(psi);

end