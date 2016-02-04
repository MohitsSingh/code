function [candidates,inds] = splitByDirection(seglist)
candidates = {};
inds = {};
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
        inds{end+1} = k;
    end    
end
end