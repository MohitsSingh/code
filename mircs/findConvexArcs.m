function candidates = findConvexArcs(seglist,E,edgelist)
%
lengths = cellfun(@(x) size(x,1),edgelist);
edgelist(lengths < 3) = [];

if (isempty(edgelist))
    candidates = {};
    return;
end

% find where edgelists meet. These are possible combinations.
%
startPoints = cellfun(@(x) x(1,:),edgelist,'UniformOutput',false);
endPoints = cellfun(@(x) x(end,:),edgelist,'UniformOutput',false);
startPoints = cat(1,startPoints{:});
endPoints = cat(1,endPoints{:});
startPoints = sub2ind2(size(E),startPoints);
endPoints = sub2ind2(size(E),endPoints);

im = edgelist2image(edgelist, size(E));

[rj, cj, re, ce] = findendsjunctions(E, 0);
% junctionPoints = sub2ind2(size(E),[rj cj]);
% junctionPoints = 
% for each junction, check to which start/end points it refers.
% imshow(im); hold on; plot(cj,rj,'g*');
im = padarray(im,[1 1],0,'both');
G = zeros(length(edgelist)); % create an edge adjacency graph :-)
aux = zeros(size(G)); % store in each entry what is the direction of the connection:
% start of e1 with end of e2, start of e1 with start of g2, etc.
for k = 1:length(rj)
    x = cj(k)+1; y = rj(k)+1;
    %     clf; imagesc(im); hold on; plot(x,y,'g*');
    %     pause
    curEdges = im(y-1:y+1,x-1:x+1);
    [xx,yy] = meshgrid(x-2:x,y-2:y);
    neighborhood = sub2ind2(size(E),[yy(:) xx(:)]);
    u = unique(curEdges(curEdges>0));
    for i1 = 1:length(u)
        p1 = [startPoints(u(i1)) endPoints(u(i1))];
        t1 = ismember(p1,neighborhood);                        
        for i2 = i1+1:length(u)
            G(u(i1),u(i2)) = 1;            
            p2 = [startPoints(u(i2)) endPoints(u(i2))];
            t2 = ismember(p2,neighborhood);
            aux(u(i1),u(i2)) = 2*t1(2)+t2(2)+1;   %a binary code: [0 0]->start1 start2, [0 1]->start1 end2, etc.
        end
    end
end

%% symmetrize the graph...
for ii = 2:size(G,2)
    for jj = ii-1:ii
        if (G(jj,ii))
            G(ii,jj) = 1;
        end
    end
end
        

% find neighboring sequences of edges
groups = enumerateGroups(G,3);
% cat(2,edgelist(groups{2}){:})
edgelist_expanded = {};
edgelist = col(edgelist);
for iGroup = 3:length(groups) % 
    curGroup = groups{iGroup};
            
    %goodCombinations = true(size(curGroup,1)); 
    %TODO: do not allow a segment to be a neighbor of two previous segments.
    % this will allow sequences of length > 2.
    
    goodCombinations = true(size(curGroup,1),1); 
    for ii = 1:size(curGroup,1)
        visited = false(size(G,1));
        gg = curGroup(ii,:);
        visited(curGroup(ii)) = true;
%         for jj = 3:length(gg)
%             G(gg(jj),:)
%         end
%         for jj = ii+2:
    end
    
    curEdges = edgelist(curGroup);
        
    for ii = 1:size(curEdges,1)
        
        curSequence = curEdges(ii,:);
        % make sure that endpoints and startpoints are consistent.
        s = {};
        s{1} = curSequence{1};
%         imagesc(E); axis image; hold on;
%                 plot(s{1}(:,2),s{1}(:,1),'g');
        prevFlipped = false(1,2);
        for iStart = 2:size(curEdges,2)
            %switch (aux(curGroup(ii,iStart),curGroup(ii,iStart+1))
            flipStatus = aux(curGroup(ii,iStart-1),curGroup(ii,iStart));
            pts = curSequence{iStart};
            t0 = s{iStart-1};
% %             pStart1 = t0(1,[2 1 ]); pEnd1 = t0(end,[2 1 ])-pStart1;
% %             pStart2 = pts(1,[2 1 ]); pEnd2 = pts(end,[2 1 ])-pStart2;
% %             
            % %             imagesc(E); axis image; hold on;
            % %             quiver(pStart1(1),pStart1(2),pEnd1(1),pEnd1(2),0,'r');
            % %             quiver(pStart2(1),pStart2(2),pEnd2(1),pEnd2(2),0,'g');
            % %
            switch flipStatus % 1:    <--.-->  2: <--<-- 3: -->--> 4: --><--
                case 1 % flip left side
                    s{iStart-1} = flipud(s{iStart-1});
                    prevFlipped(1) = true;
                case 2 % flip both sides, unless left was already flipped (so it's -->)
                    prevFlipped(2) = true;
                    pts = flipud(pts);
                    if (~prevFlipped(1))
                        s{iStart-1} = flipud(s{iStart-1});
                        prevFlipped(1) = true;
                    end
                case 3  % good
                case 4 % flip right side, unless already flipped
                    if (~prevFlipped(2))
                        pts = flipud(pts);
                        prevFlipped(2) = true;
                    end
            end
            s{iStart} =pts;
        end
        edgelist_expanded{end+1} = cat(1,s{:});
          plot(edgelist_expanded{end}(:,2),edgelist_expanded{end}(:,1),'r');
          r1 = 1;
    end
    
      
        
end

% % for k = 1:length(edgelist_expanded)
% %     clf;clf;
% %     k
% %     imagesc(E); axis image;hold on;
% % %     groups{2}(k,:)
% %     drawedgelist(edgelist_expanded(k),size(E),2,'rand');
% %     pause;
% % end
% % 

seglist = lineseg(edgelist_expanded,2); % this includes all line-segments of order 1.
% % 
% for q = 1:length(seglist)
%     q
%     clf; imagesc(E); axis image;
%     drawedgelist(seglist(q),size(E),1,'rand');
%     pause
% end

% startInds = ind2sub(size(E),startPoints);
% Z =zeros(size(E));
% Z = Z+E;
% Z(startInds) = Z(startInds)+2;
%
% imagesc(E); hold on; plot(startPoints(:,2),startPoints(:,1),'r*');
% imagesc(E);
% hold on; plot(endPoints(:,2),endPoints(:,1),'g.');

% [rj, cj, re, ce] = findendsjunctions(E, 0);
%
% E(sub2ind(size(E),rj,cj)) = 0; % break junctions
% L = bwlabel(E,4);
% %
% % [rj, cj, re, ce] = findendsjunctions(E, 0);
% % imagesc(L); hold on; plot(cj,rj,'g*')
% r = regionprops(L,'PixelList');
% %
%
% arcs = []; % search the middle column for starting candidates
% [yy,~,v] = find(L(:,round(end/2)));
%
% if (isempty(yy))
%     return;
% end

%bwtraceboundary


%
% for iY = 1:length(yy)
%     xy = r(v(iY)).PixelList;
%     xmin = min(xy(:,1));
%     xmax = max(xy(:,1));
%     x = xy(:,1);
%     y = xy(:,2);
%     %[x_,y_] = reducem(x,y);
%
%     len = floor(length(x)/5);
%     x_ = x(1:len:end);
%     y_ = y(1:len:end);
%     clf;
%     imshow(E); hold on; plot(x,y,'r-');
%     hold on; plot(x_,y_,'g-+');
%     pause;
%     % arc should be at
%
%     % check convexity
% end
%
% find

% find maximal number of consecutive segments which has same
% "direction" of turning.
% retain only edge lists that are consistent in direction
% imshow(E); hold on;
% drawedgelist(seglist,size(E),2,'rand');

% goods =false(size(seglist));


candidates = {};

for k = 1:length(seglist)
    curSegs = seglist2segs(seglist(k));
    if (size(curSegs,1) ==1)
        continue;
    end
        
%     curSegs = curSegs(:,[2 1 4 3]);
    
    vecs = segs2vecs(curSegs);
%     vc = vecs;normalize_vec(vecs')';
    crosses = zeros(size(vecs,1)-1,1);
    for kk = 1:length(crosses)
        crosses(kk) = vecs(kk,1)*vecs(kk+1,2)-vecs(kk+1,1)*vecs(kk,2);
    end
%     clf;
%     imagesc(E); hold on;axis image;
%     drawedgelist(seglist(k),size(E),2,'rand');

%      drawedgelist(candidates(k),size(E'),1,'rand');
%     drawedgelist(candidates(k),size(E'),1,'rand');
    
%     plot(curSegs(1,4),curSegs(1,3),'ms');
%     for kk = 1:size(crosses,1)    
%         if (crosses(kk)>0)
%             plot(curSegs(kk,4),curSegs(kk,3),'g+','MarkerSize',9,'LineWidth',2);
%         else
%             plot(curSegs(kk,4),curSegs(kk,3),'r+','MarkerSize',9,'LineWidth',2);
%         end
%     end

    cross_signs = sign(crosses);    
    signChanges = find(diff(cross_signs));
    
    groups = [1; signChanges+1];
    if (groups(end) < size(curSegs,1))
        groups = [groups;size(curSegs,1)];
    end
    
    for iGroup = 1:length(groups)-1
        candidates{end+1} = curSegs(groups(iGroup):groups(iGroup+1),:);
    end
%     
%     goods(k) = abs(sum(sign(crosses))) == length(crosses);
    %     for i1 = 1:length(signChanges)
%         
%     end
    
    % find groups of segments with same sign
    
    %

end
% 
% for k = 1:length(candidates)
%     clf; imagesc(E); hold on; 
%     %candidates(k)
%     sss = [candidates{k}(:,1:2);candidates{k}(end,3:4)]
%     drawedgelist({sss},size(E),2,'rand');
%     pause;
% end
% 
% imshow(E); hold on;
% drawedgelist(seglist(goods),size(E),2,'rand');
% 
% imshow(E); hold on;
% drawedgelist(seglist,size(E),1,'rand');
% % % %
% % % for k = 1:length(edgelist)
% % %     ee = edgelist{k};
% % %     %         K = convhull(ee(:,1),ee(:,2));
% % %
% % % %     clf;
% % %     imagesc(E);
% % %     hold on
% % %     x = ee(:,2);y=ee(:,1);
% % %     plot(ee(:,2),ee(:,1),'g.','LineWidth',2);
% % %
% % %     ellipse_t = fit_ellipse(x,y);
% % %     if (isempty(ellipse_t) || length(ellipse_t.status) > 0)
% % %
% % %
% % %         continue;
% % %     end
% % %     plot_ellipse(ellipse_t);
% % %     ellipse_t.status
% % %     pause;
% % % end

end