function w;
npos = size(pos_samples,2);
y = ones(npos,1);
y = [y;-ones(size(neg_samples,2),1)];
 
ss = double([pos_samples,neg_samples])';
if (nargin < 3 || isempty(C))
    C = .01;
end
if (nargin < 4 || isempty(w1))
    w1 =1;
end
if (nargin < 5 || isempty(s))
    s = 0;
end
% svm C, pos weight according to exemplarsvm, but with c = .1
svm_model = svmtrain(y, ss,sprintf(['-t %d -c' ...
    ' %f -w1 %.9f -q'], s, C, w1));

% sum support vectors with coefficients
w = svm_model.Label(1)*svm_model.SVs'*svm_model.sv_coef;
% svm_weights = (sum(svm_model.SVs .* ...
%                          repmat(svm_model.sv_coef,1, ...
%                                 size(svm_model.SVs,2)),1));

ws = w(:);
b = svm_model.rho;
sv = svm_model.SVs';
coeff = svm_model.sv_coef;