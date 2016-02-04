function c = ClosedChain(ec, f, l)

% is edgelchain ec closed ?
% true if endpoints closer than some
% fraction f of the total length l.
% If omitted, f = 0.1
% If not provided, l will be computed
% (giving it saves computation time)
%

if nargin < 2
  f = 0.1;
end

if nargin < 3
  % compute length l
  x = ec(1,:); y = ec(2,:);
  xy = [x;y]; dec = diff(xy.').';
  t = cumsum([0, sqrt([1 1]*(dec.*dec))]);    % curve length t(i) at point i (pixel)
  l = t(end);
end

p1 = ec(:,1);
p2 = ec(:,end);
if sqrt([1 1] * ((p1-p2).*(p1-p2)) ) < f*l
  c = true;
else
  c = false;
end
