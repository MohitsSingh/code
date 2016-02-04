function mlnetf = FuseMLNets(mlnet1, mlnet2)

% Fuse main-line networks mlnet1, mlnet2
% by putting all connections together in the
% output network mlnetf

mlnetf = mlnet1;

if not(length(mlnet1)==length(mlnet2))
  error('FuseMLNets: input nets of different size');
end

for mlix = 1:length(mlnet1)
  mlnetf(mlix).front = [mlnetf(mlix).front mlnet2(mlix).front];
  mlnetf(mlix).back  = [mlnetf(mlix).back  mlnet2(mlix).back ];
end
