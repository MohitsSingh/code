function D = PairwiseIPDissimilarity(des1,des2)
% Compute dissimilarity matrix D(i,j) for
% each pair of descriptor <dess(i,:),des2(j,:)>.
%
% dess,des2 : a set of descriptors, one a each row
%
% 24s if we have 1.000 descriptors dans ds1=ds2

if nargin < 2  
  % Faster
  n = size(des1,2);
  D=zeros(n,n);
  for i=1:size(des1,1)
    P = ones(n,1) * des1(i,:);
    Dx=P-P';
    D = D+Dx.*Dx;
  end
  D = sqrt(D);
else
  n1 = size(des1,2);
  n2 = size(des2,2);
  D=zeros(n2,n1);
  for i=1:size(des1,1)
    P1 = ones(n2,1) * des1(i,:);
    P2 = ones(n1,1) * des2(i,:);
    Dx=P1-P2';
    D = D+Dx.*Dx;
  end
  D = sqrt(D)';  
end

