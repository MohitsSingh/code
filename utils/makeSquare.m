function rects = makeSquare(rects,keep)

% hold on; plotBoxes2(rects(1,:)); axis equal;

if (nargin < 2)
    keep = false;
end
[height width a] = BoxSize(rects);
if (keep)
    rads = max(height,width);
    rects = inflatebbox(rects,rads,'both','true');
else
    rads = (height.^2+width.^2).^.5;
    rects = inflatebbox(rects,rads/sqrt(2),'both','true');
end

% % % % % [height width a] = BoxSize(rects);
% % % % %
% % % % % set1 = rects(height > width,:);
% % % % % [height1 width1 ~] = BoxSize(set1);
% % % % % set1 = inflatebbox(set1,[ones(size(height1)),height1./width1],'both',false);
% % % % % [height11 width11 ~] = BoxSize(set1);
% % % % %
% % % % % set2 = rects(height <= width,:);
% % % % % [height2 width2 ~] = BoxSize(set2);
% % % % % set2 = inflatebbox(set2,[width2./height,ones(size(height))],'both',false);
% % % % % [height11 width11 ~] = BoxSize(set2);
% % % % %
% % % % % rects(height > width,:) = set1;
% % % % % rects(height <= width,:) = set2;

end