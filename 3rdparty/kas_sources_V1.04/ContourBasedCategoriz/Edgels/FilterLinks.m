function links_ix = FilterLinks(ecs, links, thresh_len, thresh_mean_curv)

% Remove links between two short edgelchains.
% A link is removed if and only if both edgelchains
% are shorter than thresh_len, or if any of the two has mean curvature
% above thresh_mean_curv
%
% Output:
% links_ix = indexs of links that have been removed
%

links_ix=[];
i = 0;
for l = links
  i=i+1;
  s1 = size(ecs(l(1)).chain,2);
  s2 = size(ecs(l(2)).chain,2);
  mc1 = sum(abs(ecs(l(1)).k))/size((ecs(l(1)).chain),2);
  mc2 = sum(abs(ecs(l(2)).k))/size((ecs(l(2)).chain),2);
  if ((s1 < thresh_len) & (s2 < thresh_len)) | (mc1 > thresh_mean_curv) | (mc2 > thresh_mean_curv)
    links_ix = [links_ix i];
  end
end
