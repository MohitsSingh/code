function boundaryBenchHuman(pbRoot,pres)
% function boundaryBenchHuman(pbRoot,pres)
%
% Compute the human precision/recall data for the BSDS test images.
%
% See also imgList, bsdsRoot.
%
% David Martin <dmartin@eecs.berkeley.edu>
% March 2003

iids = imgList('test');
groundPath = fullfile('D:','symmetry_detector','symmetry_detector','Groundtruth','test2');
scorePath = 'F:\'


cR_total = 0;
sR_total = 0;
cP_total = 0;
sP_total = 0;
scores = zeros(0,4);

for i = 1:numel(iids),
  iid = iids(i);
  fprintf(2,'Processing %s image %d/%d (iid=%d)...\n',...
          pres,i,numel(iids),iid);

  fprintf(2,'  Loading groundtruth...\n');
  load(fullfile(groundPath,sprintf('groundtruth_%d',iid)),'ridge_gt')
  [h,w,k] = size(ridge_gt);
  
  fwrite(2,'  Computing PR ');
  cntR = zeros(k,1);
  sumR = zeros(k,1);
  cntP = zeros(k,1);
  sumP = zeros(k,1);
  progbar(0,k);
  for j = 1:k,
    % leave each out in turn
    pb = ridge_gt(:,:,j);
    ridge2 = zeros(h,w,k-1);
    for m = 1:k,
      if m < j, ridge2(:,:,m) = ridge_gt(:,:,m); end
      if m > j, ridge2(:,:,m-1) = ridge_gt(:,:,m); end
    end
    [thresh,cntR(j),sumR(j),cntP(j),sumP(j)] = symmetryPR(pb,ridge2);
    progbar(j,n);
  end

  fprintf(2,'  Writing results...\n');

  % pr values for each subject for each image
  
  R = cntR ./ (sumR + (sumR==0));
  P = cntP ./ (sumP + (sumP==0));
  F = fmeasure(R,P);

  fname = fullfile(pbRoot,pres,'human',sprintf('%d_pr.txt',iid));
  fid = fopen(fname,'w');
  if fid==-1, 
    error(sprintf('Could not open file %s for writing.',fname));
  end
  fprintf(fid,'%10g %10g %10g\n',[R P F]');
  fclose(fid);

  % overall pr for each image across all subjects

  cR = sum(cntR);
  sR = sum(sumR);
  cP = sum(cntP);
  sP = sum(sumP);
  bestR = cR ./ (sR + (sR==0));
  bestP = cP ./ (sP + (sP==0));
  bestF = fmeasure(bestR,bestP);
  scores(i,:) = [iid bestR bestP bestF];

  fname = fullfile(pbRoot,pres,'human','scores.txt');
  fid = fopen(fname,'w');
  if fid==-1, 
    error(sprintf('Could not open file %s for writing.',fname));
  end
  fprintf(fid,'%10d %10g %10g %10g\n',scores(1:i,:)');
  fclose(fid);

  % overall pr across all images
  
  cR_total = cR_total + cR;
  sR_total = sR_total + sR;
  cP_total = cP_total + cP;
  sP_total = sP_total + sP;
  R = cR_total ./ (sR_total + (sR_total==0));
  P = cP_total ./ (sP_total + (sP_total==0));
  F = fmeasure(R,P);
  
  fname = fullfile(pbRoot,pres,'human','score.txt');
  fid = fopen(fname,'w');
  if fid==-1, 
    error(sprintf('Could not open file %s for writing.',fname));
  end
  fprintf(fid,'%10g %10g %10g\n',R,P,F);
  fclose(fid);
end

% compute f-measure fromm recall and precision
function [f] = fmeasure(r,p)
f = 2*p.*r./(p+r+((p+r)==0));


