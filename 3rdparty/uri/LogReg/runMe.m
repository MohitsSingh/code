if ~exist('init', 'var') || init
    addpath(genpath('~/Workspace/Libs/MarkSchmidt/'));

    % sample data
    N = 1e5;
    D = 10;
    muDist = 2;
    y = rand(N, 1) > .99; % much less positives
    mu1 = rand(1, D) + muDist;
    sigma1 = rand(D);
    sigma1 = sigma1'*sigma1 / 2  + eye(D);
    
    mu2 = rand(1, D) - muDist;
    sigma2 = rand(D);
    sigma2 = sigma2'*sigma2 / 2  + eye(D);
    x = bsxfun(@times, y, mvnrnd(mu1, sigma1, N)) + bsxfun(@times, ~y, mvnrnd(mu2, sigma2, N));        
    
    init = false;
end

% training
[w1 b1] = logReg(x, y);                  % unweighted
[w2 b2] = logReg(x, y, invWeights(y));   % weighted

% testing
p1 = sigmoid(x*w1 + b1);
p2 = sigmoid(x*w2 + b2);

