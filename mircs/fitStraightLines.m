function seglist = fitStraightLines(edgelist,inds)
    seglist = {};%cell(size(edgelist));
    for k = 1:length(edgelist)
        t = inds{k};
        for u = 1:length(t)-1
            curPts = edgelist{k}(t(u):t(u+1),:);
            xy = fliplr(curPts);            
            if (size(xy,1)==2)
                seglist{end+1} = curPts;
            else
            x = xy(:,1);y = xy(:,2);
            [C, dist] = fitline(xy');
            
%             C
%             if (C(2)~=0)                                                
                y = (-C(3)-C(1)*x)/C(2);
                seglist{end+1} = fliplr([xy([1 end],1) y([1 end])]);
%             else
%                 x = (-C(3)-C(2)*xy(:,2))/C(1);
%                 seglist{end+1} = fliplr([ x([1 end]) xy([1 end],2)]);
%             end
            
            m = max(abs(seglist{end}(:)));
            if (m > 400)
                a = 1;
            end
            
            
            end
            %hold on; plotPolygons(fliplr(curPts),'g-+')
        end
    end
end