function res = arrayfun2(funer, structy)
%An alias for cellfun with UniformOutput set to false
if nargin<2
  error('Not enough arguments to cellfun2\n');
end
res = arrayfun(funer, structy, 'UniformOutput', false);