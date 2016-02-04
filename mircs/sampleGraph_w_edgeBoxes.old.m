function [boxes,routes] = sampleGraph_w_edgeBoxes(imgData,I,min_gt_ovp,gt_graph,regionSampler)
% actually, sampling a chain.


gt_boxes = cellfun3(@(x) x.bbox,gt_graph);
regionSampler.clearRoi();
regionSampler.borders = [];
configurations = {};
boxes = regionSampler.sampleEdgeBoxes(I);

% check overlap of each box with each g.t. box.
anchorBox = gt_boxes(1,:);

if (min_gt_ovp > 0)    
    gt_boxes = gt_boxes(2:end,:);
    ovp_gt = boxesOverlap(boxes,gt_boxes);
    nNodes = size(gt_boxes,1);
    candidate_boxes = struct('boxes',{});
    candidate_edges = struct('edges',{});
    node_counts = zeros(nNodes+1,1);
    for iNode = 1:nNodes
        curBoxes = boxes(ovp_gt(:,iNode) > min_gt_ovp,:);
        curBoxes = curBoxes(:,1:4);
        candidate_boxes(iNode).boxes = [gt_boxes(iNode,:);curBoxes];
        node_counts(iNode+1) = size(candidate_boxes(iNode).boxes,1);
    end
    
    big_graph = zeros(sum(node_counts));
    
    for iEdge = 1:nNodes-1
        b1 = candidate_boxes(iEdge).boxes;
        n1 = size(b1,1);
        b2 = candidate_boxes(iEdge+1).boxes;
        n2 = size(b2,1);
        G = buildBoxGraph2(b1,b2);
        [ii,jj] = find(G);
        big_graph(node_counts(iEdge)+ii,node_counts(iEdge+1)+jj) = 1;
        candidate_edges(iEdge).edges = [ii jj];
    end
    
    % sample edges to find random routes from start to end and remove
    % duplicates
    
    nRoutes = 100;
    routes = zeros(nRoutes,nNodes);
    nn = cumsum(node_counts);
    for iNode = 1:nNodes
        routes(:,iNode) = randi([nn(iNode)+1,nn(iNode+1)],nRoutes,1);
    end
    routes = unique(routes,'rows');
    goods = true(size(routes,1),1);
    for iEdge = 1:nNodes-1
        curEdges = routes(:,iEdge:iEdge+1);
        inds = sub2ind2(size2(big_graph),curEdges);
        goods = goods & big_graph(inds);
    end    
    routes =  routes(goods,:);
    boxes = [anchorBox;cat(1,candidate_boxes.boxes)];
    routes = [ones(size(routes,1),1),routes+1];
%     routes = routes-repmat(nn(1:end-1)',size(routes,1),1);
%     routes = [routes;ones(1,size(routes,2))]; % add the ground-truth    
end