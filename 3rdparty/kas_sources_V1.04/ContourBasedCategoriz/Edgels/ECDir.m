function d = ECDir(eds, l)

% direction along the edgelchain when
% going from edgel eds(1) to edgel eds(2)
%
% Output:
% d==1 -> towards the start of the chain
% d==2 -> towards the end of the chain
%
% if eds(1)==eds(2) -> selects closest endpt
%

if eds(1) > eds(2)
  d = 1;
  return;
end

if eds(1) < eds(2)
  d = 2;
  return;
end

if eds(1) == eds(2)
  if (l-eds(1)) < eds(1)
    d=2;
  else
    d=1;
  end
  return;
end
