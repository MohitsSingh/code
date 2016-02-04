function Y = vl_l2normloss(X,c,dzdy)

if nargin <= 2 
Y = 0.5sum((squeeze(X)'-c').^2);
else
assert(numel(X) == numel(C));
n = size(X,1) * size(X,2);
if nargin <= 2
  Y = sum((X(:) - c(:)).^2) ./ (2*n);
else
  assert(numel(dzdy) == 1);
  Y = reshape((dzdy / n) * (X(:) - C(:)), size(X));
end

end


%%%
assert(numel(X) == numel(c));

d = size(X);

assert(all(d == size(c)));

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    
    Y = 1 / 2 * sum(subsref((X - c) .^ 2, substruct('()', {':'}))); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = 1 / (2 * prod(d(1 : 3))) * sum(subsref((X - c) .^ 2, substruct('()', {':'}))); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    Y = dzdy * (X - c); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = dzdy / prod(d(1 : 3)) * (X - c); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
end

end
