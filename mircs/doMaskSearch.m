function R = doMaskSearch(I,featureExtractor,w,maxIters)
if nargin < 4
    maxIters = 15;
end
imRect = [1 1 fliplr(size2(I))];
curRect = imRect;
% curRect = makeSquare(imRect,true);
sz = size2(I);
nIters = 0;
R = zeros(size2(I));

scores = -inf(1,maxIters);

% do a depth first search?

beGreedy = true;


node0 = struct('rect',imRect,'mask',[],'img',[],'score',1000,'depth',0);


myChopFactor = 5;
isAbs = true;
nodes = splitNode(node0,1,myChopFactor,isAbs);


% initialize queue with first split.
Q = push([],nodes);
maxDepth = 60;
iIter = 0;
while ~isempty(Q)
    %     curNode = Q.remove();
    
    [curNode,Q] = pop(Q);
    R = R + box2Region(curNode.rect, sz, false);
    if (beGreedy)
        Q = [];
    end
    if (curNode.depth >= maxDepth)
        continue
    end
    
    if mod(iIter,5)==0
        clf; imagesc2(I); plotBoxes(curNode.rect);
        drawnow; pause(.01);
    end
    %     dpc
    %
    iIter = iIter+1;
    newNodes = splitNode(curNode,iIter,myChopFactor,isAbs);
    
    %     for t = 1:length(nodes)
    %         Q.add({nodes(t),nodes(t).score});
    %     end
    Q = push(Q,newNodes);    
end
    
    function nodes = splitNode(node,iIter,p_chop,isAbs)
        curRect = node.rect;
        if nargin < 3
            %p_chop=max(.8,1-.8.^iIter);
            p_chop=.9;
        end                
        if (1)
            right_child = chopLeft(curRect,p_chop,isAbs);
            left_child = chopRight(curRect,p_chop,isAbs);
            bottom_child = chopTop(curRect,p_chop,isAbs);
            top_child = chopBottom(curRect,p_chop,isAbs);
            rects = [right_child;left_child;bottom_child;top_child];
            %rects = inflatebbox(rects,1.1,'both',false);
            
            % also move around a bit
% %             rects2 = repmat(curRect,4,1);
% %             a = (curRect(3)-curRect(1))/3;
% %             b = (curRect(4)-curRect(2))/3;
% %             f = 0;
% %             for dx = [-a, a]
% %                 for dy = [-b, b]
% %                     f = f+1;
% %                     rects2(f,:) = rects2(f,:)+[dx dy dx dy];
% %                 end
% %             end
% %             rects2 = inflatebbox(rects2,.8,'both',false);
% %             rects = [rects;rects2];
            
        else
            rects = inflatebbox(curRect,.8,'both',false);
            a = (rects(3)-rects(1))/3;
            rects = repmat(rects,4,1);
            f = 0;
            for d1 = [-a, a]
                for d2 = [-a, a]
                    f = f+1;
                    rects(f,:) = rects(f,:)+[d1 d2 d1 d2];
                end
            end
        end
        
        rects = round(rects);
        
        nodes = struct('rect',{},'mask',{},'img',{},'score',{});
        
        
        pp = randperm(size(rects,1));
        rects = rects(pp,:);
        rects = BoxIntersection(rects,imRect);
        if (nIters>0) % keep only reasonable aspect ratios
            [a,b] = BoxSize(rects);
            aspects = max(a./b,b./a);
            rects = rects(aspects<3,:);
        end
        nIters = nIters+1;
        masked_images = {};
        for t = 1:size(rects,1)
            nodes(t).rect = rects(t,:);
            %mask = max(box2Region(rects(t,:),sz,true),curMask);
            mask = box2Region(rects(t,:),sz,true);
            %nodes(t).mask = mask;
            %nodes(t).img = I.*(1-mask)+.5*mask;
            masked_images{t} = I.*(mask)+.5*(1-mask);
            nodes(t).depth = node.depth+1;
        end
        %         masked_images = {nodes.img};
        %f = extractDNNFeats(masked_images,net_deep,layers,false);
        f = featureExtractor.extractFeaturesMulti(masked_images);
        curScores = w*f;
        for t = 1:length(nodes)
            nodes(t).score = curScores(t);
        end
        
        %         clf; imagesc2(I);
        %         plotBoxes(rects);
        %         [v,iv] = min(curScores);
        %         plotBoxes(rects(iv,:),'r-','LineWidth',3);
        %         dpc
        
    end
    function [node,Q] = pop(Q)
        % pop from presumabely sorted Q
        node = Q(end);
        Q = Q(1:end-1);
    end
    function Q = push(Q,nodes)
        Q = [Q,nodes];
        scores = [Q.score];
        [r,ir] = sort(scores,'ascend');
        Q = Q(ir);
    end
end
