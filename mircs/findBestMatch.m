function bestMatch = findBestMatch(IJ,x,myImgs,XX)
    d = l2(x',XX')';
    [dists,inds] = sort(d,1,'ascend');
    sums = sum(dists(1:3,:),1);    
   
    [v,iv] = min(sums);
    labels = zeros(size(sums));
    labels(iv) = 1;
    m = paintRule(IJ,labels,[],[],3);
%     m = mImage(cellfun2(@(x)addBorder(x,3,[0 255 0]),IJ));
    m = mImage(m);
     
    bestMatch = IJ{iv};
    
    
    figure(3),clf; 
    vl_tightsubplot(1,3,1); imagesc2(m);
    vl_tightsubplot(1,3,2); imagesc2(bestMatch);
    vl_tightsubplot(1,3,3); imagesc2(mImage(myImgs));
end
