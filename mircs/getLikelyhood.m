function [gamma_ij,llh] = getLikelyhood(pts,mu,sigmas,pi_,nComponents,...
    knownLabels)

% log_pi = log(pi_);
sigs_inv = sigmas;
sigs_det = zeros(1,nComponents);
for j = 1:nComponents
    sigs_inv(:,:,j) = inv(sigmas(:,:,j));
    sigs_det(j) = det(sigmas(:,:,j))^.5;
end

a = (2*pi)^(nComponents/2);

n = size(pts,2);
if (nargin < 6) 
    knownLabels = false(n,1);
end

d_ij = zeros(n,nComponents);
for iX = 1:n
    if (~knownLabels(iX))
        for iC = 1:nComponents
            x_d = pts(:,iX)-mu(:,iC);
            d_ij(iX,iC) = -.5*x_d'*sigs_inv(:,:,iC)*x_d;
        end
    end
end

% do calculation in log-scale
gamma_ij = zeros(n,nComponents);
for iX = 1:n
    if (~knownLabels(iX))
        for iC = 1:nComponents
            gamma_ij(iX,iC) = log(pi_(iC))+d_ij(iX,iC)-log(sigs_det(iC)*a);
        end
    end
end

T = logsumexp(gamma_ij,2);

llh = sum(T); % loglikelihood
logR = bsxfun(@minus,gamma_ij,T);
gamma_ij = exp(logR);

gamma_ij(knownLabels,1) = 1;
gamma_ij(knownLabels,2:end) = 0;

end
