function candidates = findConvexArcs2(seglist,E,edgelist)
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

G = zeros(length(edgelist));

q = l2(startPoints(:),endPoints(:))==0 | ...
l2(startPoints(:),startPoints(:))==0 | ...
l2(endPoints(:),endPoints(:))==0 | ...
l2(endPoints(:),startPoints(:))==0;
G(1:size(q,1),1:size(q,2)) =q & ~eye(size(q,1));

% find neighboring sequences of edges
groups = enumerateGroups(G,2);
% cat(2,edgelist(groups{2}){:})
edgelist_expanded = {};
edgelist = col(edgelist);
for iGroup = 1:2%length(groups) %
    curGroups = groups{iGroup};
    curEdges = edgelist(curGroups);
    for ii = 1:size(curEdges,1)
        curSequence = curEdges(ii,:);
        % make sure that endpoints and startpoints are consistent.
        s = curSequence{1};
        badSequence = false;
        for iStart = 2:size(curSequence,2)
            pts = curSequence{iStart};
            p00 = s(1,:);
            p01 = s(end,:);
            p10 = pts(1,:);
            p11 = pts(end,:);
%             
%                             clf; imagesc(E); hold on;
%                             plot(s(:,2),s(:,1),'g-');
%                             plot(pts(:,2),pts(:,1),'b-');
%                             plot(p01(2),p01(1),'m*');
%                             plot(p10(2),p10(1),'rs');
            %
            % only acceptable condition is that previous end is current
            % beginning. otherwise flip current.
            
            if all(p01==p10) % -->--> do nothing
            elseif all(p00==p11) % <-- <-- % do nothing
            elseif all(p01==p11) % --> <-- flip end
                pts = flipud(pts);
            elseif all(p00==p10) % <-- --> % flip start
                s = flipud(s);
            else % a t-junction.
                badSequence = true;
            end
            s = [s;pts];
        end
        
        %
        %         plot(s(:,2),s(:,1),'g');
        %         if (badSequence)
        %             plot(s(:,2),s(:,1),'r');
        % %
        %         end
        if (~badSequence)
            edgelist_expanded{end+1} = s;
            imagesc(E); axis image; hold on;
            plot(edgelist_expanded{end}(:,2),edgelist_expanded{end}(:,1),'r');
            r1 = 1;
        end
        %         end
    end
end


% remove duplicates....
lengths = cellfun(@length,edgelist_expanded);
uu = unique(lengths);
edgelist_new = {};
for k = 1:length(uu)
    x1 = edgelist_expanded(lengths==uu(k));
    x = cell2mat(cellfun2(@(x) col(x),x1))';
    [x,ia] = unique(x,'rows');
    edgelist_new = [edgelist_new,x1(ia)];
end
edgelist_expanded = edgelist_new;

seglist = lineseg(edgelist_expanded,1);

candidates = {};

for k = 1:length(seglist)
    curSegs = seglist2segs(seglist(k));
    if (size(curSegs,1) ==1)
        continue;
    end
       
    
    vecs = segs2vecs(curSegs);
    
    crosses = zeros(size(vecs,1)-1,1);
    for kk = 1:length(crosses)
        crosses(kk) = vecs(kk,1)*vecs(kk+1,2)-vecs(kk+1,1)*vecs(kk,2);
    end   
    
    cross_signs = sign(crosses);
    signChanges = find(diff(cross_signs));
    
    groups = [1; signChanges+1];
    if (groups(end) < size(curSegs,1))
        groups = [groups;size(curSegs,1)];
    end
    
    for iGroup = 1:length(groups)-1
        candidates{end+1} = curSegs(groups(iGroup):groups(iGroup+1),:);
    end   
end

% remove duplicate candidates

end