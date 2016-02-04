function [EP, pas_ids] = ComputeEdgelsOnkAS_LUT(I, P, bb)
    vak=Value_of_vak(size(I.pas,1));
% compute lookup table EP, so that EP(pas_ix).eixs = indeces in P of edgels on pas I.pas(:,pas_ix)
% bb = bounding-box of P
%

if isempty(I.pas)
  EP = []; pas_ids = [];
  return;
end

% prepare edgel-index table: given edgel coords, return index in P
max_P = max(P');
Eix = uint16(zeros(max_P+1));
ix = 0;
for ed = P
  ix = ix+1;
  Eix(ed(1)+1,ed(2)+1) = ix;
end

% prepare lookup table telling which edgels are covered by a pas
pas_ids = PointsInBB(I.pas_ls(1:2,:), bb);                     % pas with center inside bb
EP(length(pas_ids)).eixs = uint16([]);                         % index in P
for pas_ix = 1:length(pas_ids)
  temp = GetEdgelsUnderMLs(I, I.pas(1:vak, pas_ids(pas_ix))');
  EP(pas_ix).eixs = zeros(1,size(temp,2));
  for i = 1:size(temp,2)
    if temp(1,i) <= max_P(1) && temp(2,i) <= max_P(2) 
      EP(pas_ix).eixs(i) = Eix(temp(1,i)+1, temp(2,i)+1);
    end
  end
  EP(pas_ix).eixs = EP(pas_ix).eixs(find(EP(pas_ix).eixs));    % keep only edgels inside bb
end
