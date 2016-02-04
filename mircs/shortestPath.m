function [ shortestPathImage ] = shortestPath(pixelCost,I,xy)
%SHORTESTPATH Summary of this function goes here
%   Detailed explanation goes here
x = xy(1);
y = xy(2);
% T = .15;
% Z = ucm > T;
% D_edge = bwdist(Z);
% pixelCost = (1-ucm);
% pixelCost(ucm<=T) = D_edge(ucm<=T);
% make a pixel graph.
Z = zeros(size(pixelCost));
[X,Y] = meshgrid(1:size(Z,2),1:size(Z,1));
inds = sub2ind(size(Z),Y,X);
m = numel(inds);
M = sparse(m,m);
sz = size(Z);

% discard all border indices for simplicity.
inds_right = sub2ind(sz,Y(2:end-1,2:end-1),X(2:end-1,2:end-1)+1);
inds_bottom = sub2ind(sz,Y(2:end-1,2:end-1)+1,X(2:end-1,2:end-1));
inds_bottomright = sub2ind(sz,Y(2:end-1,2:end-1)+1,X(2:end-1,2:end-1)+1);
inds_bottomleft = sub2ind(sz,Y(2:end-1,2:end-1)+1,X(2:end-1,2:end-1)-1);
inds_ = inds(2:end-1,2:end-1);

transition_side = 1;
transition_diag = sqrt(2);

%      costs_right = min(pixelCost(inds_),pixelCost(inds_right))+transition_side;
%      costs_bottom = min(pixelCost(inds_),pixelCost(inds_bottom))+transition_side;
%      costs_bottomright = min(pixelCost(inds_),pixelCost(inds_bottomright))+transition_diag;
%      costs_bottomleft = min(pixelCost(inds_),pixelCost(inds_bottomleft))+transition_diag;


costs_right = pixelCost(inds_right)+transition_side;
costs_bottom = pixelCost(inds_bottom)+transition_side;
costs_bottomright = pixelCost(inds_bottomright)+transition_diag;
costs_bottomleft = pixelCost(inds_bottomleft)+transition_diag;

M = max(M,sparse(inds_,inds_right,costs_right,m,m));
M = max(M,sparse(inds_,inds_bottom,costs_bottom,m,m));
% go only to 4  neighbors
M = max(M,sparse(inds_,inds_bottomright,costs_bottomright,m,m));
M = max(M,sparse(inds_,inds_bottomleft,costs_bottomleft,m,m));
M = max(M,M');

[Y_,X_] = ind2sub(size(Z),inds_);
%      imagesc(pixelCost);

%      gplot(M,[X(:),Y(:)]);
% x = 168;
% y = 173;
% figure(1); clf; imshow(I); [x,y] = ginput(1);
sourceV = inds(round(y),round(x));


% check some routes





%     figure(1),imshow(f2)
[dist,path] = dijkstra( M, sourceV );

%     [dist, path, pred] = graphshortestpath(M,sourceV,'Directed',false);

shortestPathImage = reshape(dist,size(Z));
% figure(1);clf;imagesc(exp(-shortestPathImage/100));axis image;
% figure(2);imagesc(I); axis image;
%         figure(1);clf(1);imagesc(shortestPathImage);

%     figure(2);clf(2);,imagesc(f2);

%     figure,imagesc(pixelCost)

%     clf,figure(1);hist(exp(-shortestPathImage(:)/10000));



end

