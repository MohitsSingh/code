function mls = NML(mlnet, mlix, ept)

% directly connected mls to endpoint ept to mlix in mlnet
% (basically just a wrapper around the heavy syntax of .front, .back

mls = [];
if ept == 1
  if isempty(mlnet(mlix).back) return; end
  mls = mlnet(mlix).back(1,:);  
elseif ept == 2
  if isempty(mlnet(mlix).front) return; end
  mls = mlnet(mlix).front(1,:);
else
  error('NML: ept must be either 1 or 2');
end

