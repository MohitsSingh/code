function [mu,sigmas,pi_] = emAlgorithm2(pts,labels,nComponents,Z)
% This runs the EM algorithm, where some of the points have hard
% assignments (i.e, we know they all come from the same distribution).
% pts is D x n  for D dimensional data.

% (start with a gaussian mixture model).

dim = size(pts,1);
n = size(pts,2);
knownLabels = labels~=-1;

% initialize the mu as the means of known components. Initialize the others
% using k-means.
mu = zeros(dim,nComponents);
mu(:,1) = mean(pts(:,knownLabels),2);
[C,A] = vl_kmeans(pts(:,~knownLabels),nComponents-1);
mu(:,2:end) = C; % means


sigmas = repmat(eye(dim),[1 1 nComponents]); % cov-matrices
pi_ = ones(1,nComponents)/nComponents;% mixture components

% actually, the mixture components

maxIterations = 20;
curLikelyhood = 0;
minDeltaLikelyhood = 5;
curLikelyhood = getLikelyhood(pts,mu,sigmas,pi_,nComponents);
for k = 1:maxIterations
    
    % calc credits
    % p(y_i) = p_i;
    % calculate for each X the chance for coming from each components.
    
    % (1/|det(sigma_j)|^.5* (2*pi)^k/2(x_i-m_j)'inv(sigma_j)(x_i-m_j)
    
    % invert all the sigmas, calc determinants
    sigs_inv = sigmas;
    sigs_det = zeros(1,nComponents);
    for j = 1:nComponents
        sigs_inv(:,:,j) = inv(sigmas(:,:,j));
        sigs_det(j) = det(sigmas(:,:,j));
    end
    % silly, but I'll vectorize it later
    a = (2*pi)^nComponents/2;
    P_i_j = zeros(n,nComponents); %non-normalized
    for iX = 1:n
        for iC = 1:nComponents
            x_d = pts(:,iX)-mu(:,iC);
            P_i_j(iX,iC) = pi_(iC)*exp(-.5*x_d'*sigs_inv(:,:,iC)*x_d)/(sigs_det(iC)*a);
        end
    end
    
        
    gamma_ij = bsxfun(@rdivide,P_i_j,sum(P_i_j,2));
    % set the gamma for the known components to be a delta function.
    gamma_ij(knownLabels,1) = 1;
    gamma_ij(knownLabels,2:end) = 0;
    
    % now that we have the credit assignment, we can maximize the
    % expecation.
    
    pi_ = sum(gamma_ij)/n;
    
    g_n = sum(gamma_ij);
    mu = pts*(bsxfun(@rdivide,gamma_ij,g_n));
    for iComp = 1:nComponents
        d_ = bsxfun(@minus,pts,mu(:,iComp));
        sig_ = zeros(nComponents);
        for iX = 1:n
            sig_ = sig_ + gamma_ij(iX,iComp)*d_(:,iX)*d_(:,iX)';
        end
        sig_ = sig_/g_n(iComp);
        sigmas(:,:,iComp) = sig_;
    end
    
    % finally, calculate the resulting likelyhood
    curLikelyhood = getLikelyhood(pts,mu,sigmas,pi_,nComponents);
    
   
    clf;
    figure(1);
    imagesc(Z); hold on;
    plot(pts(1,:),pts(2,:),'r.');
    plot(mu(1,1),mu(2,1),'r+');
    plot(mu(1,2),mu(2,2),'g+');
    title(num2str(curLikelyhood));
    pause(.1);
    
    
end
end


function L = getLikelyhood(pts,mu,sigmas,pi_,nComponents)
% return the log likelyhood
log_pi = log(pi_);
sigs_inv = sigmas;
sigs_det = zeros(1,nComponents);
for j = 1:nComponents
    sigs_inv(:,:,j) = inv(sigmas(:,:,j));
    sigs_det(j) = det(sigmas(:,:,j));
end
n = size(pts,2);
% calc the per-datapoint log-likelyhood for each component

a = (2*pi)^nComponents/2;
L_i_j = zeros(n,nComponents); %non-normalized
for iX = 1:n
    for iC = 1:nComponents
        x_d = pts(:,iX)-mu(:,iC);
        L_i_j(iX,iC) = log_pi(iC)-log(sigs_det(iC)*a) -.5*x_d'*sigs_inv(:,:,iC)*x_d;
    end
end

L = sum(L_i_j(:));

end