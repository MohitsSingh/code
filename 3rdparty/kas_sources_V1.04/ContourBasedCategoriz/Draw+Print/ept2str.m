function s = ept2str(e)

% convert endpt code e to word s

if e == 1
  s = 'back';
elseif e == 2
  s = 'front';
else
  s = 'wrong enpoint code';
end
