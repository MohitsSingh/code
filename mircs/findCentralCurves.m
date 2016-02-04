function centralCurves = findCentralCurves(E,segs)
% send "rays" through image.
% where rays intersect, start looking at symmetric continuation.
centralCurves = 0;
% maybe do this with a log-polar coordinate system.? anyway:

%     [y,x,vals] = find(E);
%     [vals,ivals] = sort(vals,'descend');
%     visited = false(size(E));
imCenter = fliplr(size(E))/2;
segs = segs(:,[2 1 4 3]); % from ij to xy
vecs = segs2vecs(segs);

vecs_n = normalize_vec(vecs')';
% approximate the vector to the segment by the vector to it's center.
seg_centers = (segs(:,3:4)+segs(:,1:2))/2;



% imCenter = repmat(imCenter,size(segs,1),1);

% find the angles close to 90 degrees relative to vector to center.
c_vec = bsxfun(@minus,seg_centers,imCenter);
c_vec_n = normalize_vec(c_vec')';

dots = sum(c_vec_n.*vecs_n,2);

imCenter_v = repmat(imCenter,size(segs,1),1);
% clf,imagesc(E);
drawedgelist(segs2seglist(segs(:,[2 1 4 3])),size(E),2,'g');hold on;
plot(imCenter(1),imCenter(2),'r+');
plot(seg_centers(:,1),seg_centers(:,2),'ms');
quiver(imCenter_v(:,1),imCenter_v(:,2),c_vec(:,1),c_vec(:,2),0,'b');
sel_ = abs(dots)<.5;
quiver(imCenter_v(sel_,1),imCenter_v(sel_,2),c_vec(sel_,1),c_vec(sel_,2),0,'r');



%
%
%     for k = 1:length(x)
%         curPoint = [x(k) y(k)];
%         v = curPoint-imCenter;
%         v_n = normalize_vec(v);
%         u_n = [-v_n(2) v_n(1)];
%
%         % find neighbors to the left and right of v
%
%     end


end