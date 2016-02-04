function yhat = detect(param,model,x)

yhat = constraintCB(param,model,x);
return;
nPts = param.nPts;

% y = reshape(y,nPts,2);
cellSize = param.cellSize;
windowSize_c = param.windowSize/cellSize;
imgSize_c = param.imgSize/cellSize;
nEdges = param.nEdges;
w_diffs = model.w(end-nEdges+1:end);
ww = model.w(1:end-nEdges);
ww = reshape(ww,[],nPts);
w_app =  ww(1:end-4,:);
w_loc = ww(end-4+1:end,:);
mean_locs = param.mean_labels;
% return;
yhat = zeros(nPts,2);

% return;
if (length(size(x))==2)
    x = cat(3,x,x,x);
end
wSize = [windowSize_c windowSize_c];
[X,boxes ] = all_hog_features_dumb2(param.windowSize, cellSize,single(x));
boxes = boxes(:,1:4);
box_centers = boxCenters(boxes);
goods = inImageBounds(x,box_centers);
boxes = boxes(goods,:);
box_centers = box_centers(goods,:);
X = X(:,goods);

% figure(2); clf; imagesc2(x); plotPolygons(boxCenters(boxes),'r+');

app_scores = (w_app'*X)';
box_centers_n = box_centers/size(x,1);
loc_scores = zeros(size(box_centers,1),nPts);
for u = 1:nPts
    curLocFeats = bsxfun(@minus,box_centers_n,mean_locs(u,:));
    curLocFeats = [curLocFeats,curLocFeats.^2];
    loc_scores(:,u) = curLocFeats*w_loc(:,u);
end
%loc_feats = box_centers-
%loc_scores = pts_w.*
% xx = box_centers(:,1)/param.imgSize;
% yy = box_centers(:,2)/param.imgSize;

% boxesOverlap
total_scores = app_scores+loc_scores;

%%
%% calculate pairwize scores.....
nLocs = length(total_scores);
pairwise_scores = zeros(nLocs,nLocs,param.nEdges);
% % loc_vectors = zeros(nLocs,nLocs,4);
% % % keep it simple, altough clearly vectorizable
% % % [u,v] = meshgrid(box_centers_n(:,1),box_centers_n(:,2));
% % for t1 = 1:nLocs
% %     for t2 = 1:nLocs
% %         b = box_centers_n(t1,:)-box_centers_n(t2,:);
% %         loc_vectors(t1,t2,:) = [b b.^2];
% %     end
% % end
p_distances = l2(box_centers_n,box_centers_n).^.5;
adj = param.adj;
[ii,jj] = find(adj);
for iu = 1:length(ii)
    i1 = ii(iu);
    j1 = jj(iu);
%     curWeights = w_diffs( ((iu-1)*4+1):4*iu);
% %     pairwise_scores(:,:,iu) = ...
% %         sum(loc_vectors.*repmat(reshape(curWeights,[1 1 4]),size(loc_vectors,1),size(loc_vectors,2),1),3);
% for t = 1:nPts-1
%    pairwise_scores(:,:,t) = param.means( p_distances;%*w_diffs(t);
    pairwise_scores(:,:,iu) = w_diffs(iu)*exp(-(param.means(i1,j1)-p_distances).^2/(2*(param.stds(i1,j1)^2)));
end
adj = full(adj)>0;
adj = adj | adj';
% pairwise_scores = zeros(nLocs);
best_configuration = decode_structure(adj,1+total_scores,pairwise_scores+eps);
im = best_configuration;
%%


% pairwise_scores = zeros(length(total_scores));
% best_configuration = decode_structure(exp(total_scores),pairwise_scores+eps);
% im = best_configuration;

% [m,im] = max(total_scores,[],1);
yhat = [box_centers_n(im,:)];
