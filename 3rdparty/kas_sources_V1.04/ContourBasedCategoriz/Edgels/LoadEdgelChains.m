function ecs = LoadEdgelChains(fname)

% Load edgel chains from file fname,
% which has format
% ni  xi1 yi1 gi1 oi1  ...  xin yin gin oin
% ...
%
% ni = number of edgels in the ith chain
% xie,yie = coords of edgel e in chain i
% gie,oie = gradient magnitude and orientation of edgel e in chain i
%
% Output has the form
% ecs(i).chain = 2xni matrix of coords  (gie,oie are ignored)
%

fid = fopen(fname);
needmore = true;
i = 0;
ecs = [];
while needmore
  l = fgetl(fid);
  if ischar(l)
     i = i + 1;
     l = str2num(l);                     % new chain
     n = l(1);                           % number of edgels in the chain
     chain = reshape(l(2:end), 4, n)';   % edgels in [x y g o] format
     ecs(i).chain = chain(:,1:2)';       % keep only [x y]
  else
     needmore = false;
  end
end
fclose(fid);
