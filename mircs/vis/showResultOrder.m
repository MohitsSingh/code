function [true_imgs,res,res_only_sel,imgs_sorted] = showResultOrder(ims,scores,labels,sel_,maxIms)

% find where each labels appears in the final scoring.
[r,ir] = sort(scores,'descend');

if (nargin < 5)
    maxIms = inf;
end
ir = ir(1:min(maxIms,length(ir)));

labels = labels(ir); sel_ = sel_(ir);
ims = ims(ir);
if (nargout > 3)
    imgs_sorted = paintRule(ims,labels,[0 255 0],[255 0 0],2);
    imgs_sorted = multiImage(imgs_sorted);
end
true_imgs = ims(labels); % real positive images.
true_imgs = paintRule(true_imgs,sel_(labels),[0 255 0],[255 0 0],2);
% find locations of true detections within all detections.
labels_f = find(labels);
res = multiImage(true_imgs,labels_f);
res_only_sel = multiImage(true_imgs(sel_(labels)),labels_f(sel_(labels)));

%  showResultOrder(imgs,scores,labels,sel)

