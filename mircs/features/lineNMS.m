function [pick] = lineNMS(lines_,T)
%LINENMS Summary of this function goes here
%   Detailed explanation goes here

edges = createEdge(lines_(:,1:2),lines_(:,3:4));

if (nargin < 2)
    T = 5;
end

% find the distance between all pairs of edges.

% create convex hull between all pairs of edges...
P1 = lines_(:,1:2);
P2 = lines_(:,3:4);

% find the distance from each point to each edge, and build and adjacency
% matrix accordingly.


G = zeros(size(P1,1));
for k = 1:size(P1,1)
    k
    %                
    p1 = repmat(P1(k,:),size(P1,1),1);
    p2 = repmat(P2(k,:),size(P2,1),1);
    dist_1 = distancePointEdge(p1,edges);
    dist_2 = distancePointEdge(p2,edges);        
    % distance from line 1 to all lines.    
    G(k,:) = max(dist_1,dist_2);
end
G = (max(G,G'));
% G(0<eye(size(G,1))) = inf;
%[ii,jj] = find(G < 1);

% suppress redundant lines.
removed = false(size(G,1),1);
G_ = G;
G_(0<eye(size(G,1))) = inf;
for k = 1:size(G,1)
    if (removed(k))
        continue;
    end
    n = find(G_(k,:)< T);
    removed(n) = true;    
end

pick = find(~removed);

% % choose for each group the mean line.
% meanLines = zeros(length(sel_),4);
% for ik = 1:length(sel_)
%     k = sel_(ik);
%     jj = find(G(k,:)<T); % will always include at least line k
%     % find the mean of all lines here. 
%     % group all p1 points and p2 points together
% %     if (isempty(jj))
% %         meanLines(k) = lines_(k,:);
% %     end
%     d1 = l2(P1(k,:), P1(jj,:));
%     d2 = l2(P1(k,:), P2(jj,:));
%     lineGroup = [lines_(jj(d1 <= d2),:);lines_(jj(d1 > d2),[3 4 1 2])];
%     meanLines(ik,:) = mean(lineGroup,1);    
% end

% figure(1); clf; hold on; plot_svg(lines_,[],[]);    
% figure(2); clf; hold on; plot_svg(meanLines,[],[]);    
