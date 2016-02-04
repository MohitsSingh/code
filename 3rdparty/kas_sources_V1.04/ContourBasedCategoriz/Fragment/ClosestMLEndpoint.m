function ept = ClosestMLEndpoint(ml, e)

% Closest endpoint of ml to point e.
% ept = 2 -> front, ept = 1 -> back.
%

ef = MLEndpoint(ml, 2);
eb = MLEndpoint(ml, 1);
if sum((ef-e).^2) < sum((eb-e).^2)
  ept = 2;
else
  ept = 1;
end
