function weights = invWeights(y)
y = y > 0;
weights = y ./ fixdiv(sum(y)) + ~y ./ fixdiv(sum(~y));
