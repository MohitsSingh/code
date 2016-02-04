function idx = argmin(x,varargin)
% find the index of the maximum and breaks ties at random
% ties are broken by adding noise that is distributed N(0, epsi)
% [ig idx] = max(x+randn(size(x))*eps,varargin{:});
epsi = 1e-10;
[~,idx] = min(x+randn(size(x))*epsi,[],varargin{:});
