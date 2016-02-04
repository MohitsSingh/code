function psi = featureCB(param, x, y)
% disp('feature');

nPts = param.nPts;
nEdges = param.nEdges;
cellSize = param.cellSize;
if (param.precompute_responses)
    x = x{1};
end
if (length(size(x))==2)
    x{1} = cat(3,x,x,x);
end

u = y(:,1:2)*param.imgSize;
sub_rect = [u u];
sub_rect = inflatebbox(sub_rect,param.windowSize,'both',true);
subImgs = multiCrop2(x,round(sub_rect));
%%
X = getImageStackHOG(subImgs,param.windowSize,true,false,cellSize);
X = sparse(double(X));

% if param.use_location_prior
%     psi_loc = zeros(size(nPts))
% end

psi = zeros(nPts+nEdges,1);
for t = 1:nPts
    psi(t)=param.patchDetectors(t,:)*X(:,t);
end

if (param.use_location_prior)
    for t = 1:nPts
        psi(t)=psi(t)+log_lh(param.gaussian_models{t},y(t,1:2));
    end
end

for iEdge = 1:nEdges
    %     iu
    iPair = param.edges(iEdge,1);
    jPair = param.edges(iEdge,2);
    cur_offset = squeeze(y(jPair,:)-y(iPair,:));
    psi(nPts+iEdge)= log_lh(param.pair_gaussian_models{iEdge},cur_offset(1:2));
end

psi = sparse(psi);


end