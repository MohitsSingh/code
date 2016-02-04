boxDists = boxDistances(boxes)
    box2Pts(boxes(1,:))
    
    all_pts = {};
    all_inds = {};
    all_edges = {};
    for t = 1:size(boxes,1)
        curPts = box2Pts(boxes(t,:));
        curEdges = [curPts([1 2 3 4],:) curPts([2 3 4 1],:)];
        curInds = ones(4,1)*t;
        
        all_pts{t} = curPts;
        all_edges{t} = curEdges;
        all_inds{t} = curInds;
        
%         plotSegs(curEdges); hold on; plotPolygons(curPts,'r--');
%         plotBoxes(boxes(t,:),'g-:')
    end
    
    all_inds = cat(1,all_inds{:});
    all_edges = cat(1,all_edges{:});
    all_pts = cat(1,all_pts{:});
    nBoxes = size(boxes,1);
    nPts = size(all_pts,1);
    ptsToEdgesDists = zeros(nPts);
    
    for t = 1:nPts
%         t
        ptsToEdgesDists(:,t) = distancePointEdge(all_pts,all_edges(t,:));
    end
    
    box2BoxDists = zeros(
    
%     imagesc2(ptsToEdgesDists)
    
%     
%     z = distancePointEdge(all_pts(1,:),all_edges);
%     [r,ir] = sort(z,'descend');
%     for it = 1:length(z)
%         k = ir(it);
%         clf; imagesc2(I);
%         plotPolygons(all_pts(1,:),'r+','LineWidth',4);
%         plotSegs(all_edges(k,:),'g-','LineWidth',2);
%         dpc(.01);
%     end
%     
    
    
    figure(1);hold on;plotPolygons(box2Pts(boxes(1,:)));
    plotBoxes(boxes(1,:),'g--');