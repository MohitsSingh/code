function f = mySearchFun(Rs,cur_t,weights)
for k = 1:length(Rs)    
    R = Rs{k};
    if (isempty(R))
        myScores(k) = -10;
        continue;
    end
    contourLengths = R(1,:);    
    areas = R(8,:);
    inLips = R(9,:);
    ucmStrengths = R(2,:);
    t1 = contourLengths>.3;
    horzDist = R(10,:);
%     t2 = areas>.1 & areas < .2;
    t2 = horzDist;
%     curScores = t1+uecmStrengths+1*t2+inLips;
    
    
    curScores = R'*weights;
    
%     curScores(isnan(curScores)) = -100;
        
    myScores(k) = max(curScores);    
end
myScores(isnan(myScores)) = -1000;
% [v,iv] = sort(myScores,'descend');
[prec,rec,aps,T] = calc_aps2(myScores'+.01*rand(size(myScores')),cur_t);
f = -aps;
end