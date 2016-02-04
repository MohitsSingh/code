function G = buildBoxGraph(boxes,boxMargin,I)
% find out which boxes are touching each other.
if nargin < 2
    boxMargin = 3;
    %         boxMargin = .1;
end

boxes_bigger = dilateBoxes(boxes,boxMargin);
boxes_smaller = dilateBoxes(boxes,-boxMargin);
% profile on
ovp = boxesHaveIntersection(boxes);
% profile viewer
ovp_big = boxesHaveIntersection(boxes_bigger);
ovp_small = boxesHaveIntersection(boxes_smaller);
G = (ovp_big & ~ovp) | (ovp & ~ovp_small);

return;
%%%boxDists = boxDistances(boxes); % find distances between bounding boxes...
%
%     a = [1 1 10 100];
%     clf; hold on;
%     plotBoxes(a);
%     plotBoxes(inflatebbox(a,1.1,'both',false),'r-');

%     boxes_bigger = inflatebbox(boxes,1+boxMargin,'both',false);
boxes_bigger = boxes;
boxes_bigger(:,1:2) = boxes_bigger(:,1:2)-boxMargin;
boxes_bigger(:,3:4) = boxes_bigger(:,3:4)+boxMargin;
boxes_smaller = boxes;
%     boxes_smaller = inflatebbox(boxes,1-boxMargin,'both',false);
boxes_smaller(:,1:2) = boxes_smaller(:,1:2)+boxMargin;
boxes_smaller(:,3:4) = boxes_smaller(:,3:4)-boxMargin;
% ovp_big = boxesOverlap(boxes_bigger) > 0;

% [ii,jj] = find(ovp_big); % no need to check for others.....

ovp = boxesHaveIntersection(boxes);
ovp_big = boxesHaveIntersection(boxes_bigger);
ovp_small = boxesHaveIntersection(boxes_smaller);

% % intersections_small = BoxIntersection(boxes(ii,:),boxes(jj,:));
% ovp_small = boxesOverlap(boxes_smaller) > 0;
% ovp = boxesOverlap(boxes) > 0;
centers = boxCenters(boxes);
G = (ovp_big & ~ovp) | (ovp & ~ovp_small);
% [ii,jj] = meshgrid(1:size(G,1),1:size(G,1));
% G(ii < jj) = 0;
% % %
% plot the centers...
%     x2(I);
%     gplot(G,centers);
%     [ii,jj] = find(G);
%     p = randperm(length(ii));
%     p = 1:length(ii);

doDebugging  = false;
if nargin == 3
    doDebugging = true;
end
if (doDebugging)
    %%%%%clf; imagesc2(I); bb1 = getSingleRect(true);
    bb1 =  [222.6255  194.0348  248.2643  205.5933];
    ovps1 = boxesOverlap(boxes,bb1);
    
    %     showSortedBoxes(I,boxes,ovps1)
    ff1 = find(ovps1>.3);
    bb2 = [244.2523  194.1755  300.9356  242.9168];
    %     clf; imagesc2(I); bb2 = getSingleRect(true);
    ovps2 = boxesOverlap(boxes,bb2);
    ff2 = find(ovps2 > .3);
    %      showSortedBoxes(I,boxes,ovps2)
    GG = zeros(size(G));
    GG(ff1,ff2) = G(ff1,ff2);
    
    [ii,jj] = find(GG);
    %     showSortedBoxes(I,boxes,ovps2);
    for k = 1:length(ii)
        %         k = p(ip);
        e1 = ii(k);
        
        %         if ~ismember(e1,ff)
        %             continue
        %         end
        e2 = jj(k);
        
        clf; imagesc2(I);
        plotBoxes(boxes([e1 e2],:));
        dpc
    end
end
