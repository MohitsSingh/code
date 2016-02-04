function [boxes,bads] = clip_to_image(boxes,imbb)
%clip boxes to image (just changes the max dimensions)
if (numel(imbb)~=4)
    imbb = [1 1 size(imbb,2),size(imbb,1)];
end
if size(boxes,1) == 0
  return;
end
% boxes_orig = boxes;
bads = false(size(boxes,1),1);
for i = 1:2
  boxes(:,i) = max(imbb(i),boxes(:,i));
  bads = bads | boxes(:,i) > imbb(i+2);
end

for i = 3:4
  boxes(:,i) = min(imbb(i),boxes(:,i));
  bads = bads | boxes(:,i) < imbb(i-2);
end

% f = find(sum(abs(boxes_orig-boxes),2));

% boxes_orig(f,:)
