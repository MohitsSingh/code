function FB = FbMakegabor( r, nOrient, nScales, lambda, sigma )
% multi-scale even/odd gabor filters. Adapted from code by Serge Belongie.
cnt=1;
for m=1:nScales
  for n=1:nOrient
    [F1,F2]=filterGabor2d(r,sigma^m,lambda,180*(n-1)/nOrient);
    if(m==1 && n==1); FB=repmat(F1,[1 1 nScales*nOrient*2]); end
    FB(:,:,cnt)=F1;  cnt=cnt+1;   FB(:,:,cnt)=F2;  cnt=cnt+1;
  end
end