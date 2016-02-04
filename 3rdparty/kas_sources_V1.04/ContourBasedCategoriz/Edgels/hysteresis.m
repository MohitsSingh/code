function pb = hysteresis(pbin, tlow, thigh)

[r,c] = find(pbin>=thigh);
if isempty(r)
  pb = zeros(size(pbin));
  return;
end
b = bwselect(pbin>tlow,c,r,8);
pb = pbin.*b;
