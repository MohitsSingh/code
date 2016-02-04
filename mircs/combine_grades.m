function combined = combine_grades(dets1,det2,theta,gt_labels)
% combine_grades(dets1,dets2) adds to each detection
% in set set of detector results dets1 the corresponding grade from the
% detector det2 applied to the same image, with a weighting factor of
% theta. if not given, theta is set to 1.
if (nargin < 3)
    theta = 1;
end

if (isscalar(theta))
    theta = [1 theta];
end

combined = initClusters;
cLocs = det2.cluster_locs;
nDets = size(cLocs,1);
for k = 1:length(dets1)
    curLocs = dets1(k).cluster_locs;
    s = curLocs(:,12);
    ii = curLocs(:,11);
    curLocs(:,12) = theta(1)*s + theta(2)*cLocs(ii,12);
           
% %     if (~isempty(gt_labels))
% % %            [w b ap] = checkSVM( [s,cLocs(ii,12)],gt_labels(ii))
% %         plot(s(~gt_labels),cLocs(ii(~gt_labels),12),'g*')
% %         hold on;
% %         plot(s(gt_labels),cLocs(ii(gt_labels),12),'r+')
% %        w = -[1 -1]; b =0;
% %         plot(w(1)*(s),w(2)*cLocs(ii,12),'m+');
% %         %
% % %           curLocs(:,12) = [s,cLocs(ii,12)]*w;
% %     end
    
%     [~,ii] = sort(curLocs(:,12),'descend');
%     curLocs = curLocs(ii,:);
    combined(k).isvalid = true;
    combined(k).w = dets1(k).w;
    combined(k).cluster_locs = curLocs;
end
