function strengths = ComputeMLStengths(obj, S)

% Computes average edge-strength for each main-line obj.mainlines(:,i).
% S(x,y) in [0,1] = strenght of edgel (x,y)
%

if size(obj.mainlines,2) == 0
  strengths = [];
  return;
end

strengths = zeros(1,obj.mainlines(1,end));
for ml = obj.mainlines
  pts = GetEdgelsUnderMLs(obj, ml(1), false);
  pts = pts+1;   % first coord of S is (1,1), while first coord of edgel-chain is (0,0)
  strengths(ml(1)) = mean(S((pts(1,:)-1)*size(S,1)+pts(2,:)));
end
