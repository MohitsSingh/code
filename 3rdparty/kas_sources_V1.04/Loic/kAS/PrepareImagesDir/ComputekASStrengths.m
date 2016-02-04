function pas_str = ComputekASStrengths(pas, str)    
    vak=Value_of_vak(size(pas,1));
% Compute the average edge strength of each pas(:,i),
% as the average strength mean(str(pas(1:vak,i))) of its
% component segments pas(1:vak,i).
%

pas_str = [];
if isempty(pas)
  return;
end

if vak==1
    pas_str = str(pas(1:vak,:));
else
    pas_str = mean(str(pas(1:vak,:)));
end
    
