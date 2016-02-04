function aa = cutOutSubWindows(IsTr,sub_windows)
    aa = {};
    for curKP = 1:size(sub_windows,2)
        curKP
        curSubWindows = squeeze(sub_windows(:,curKP,:));        
        a = col(multiCrop([],IsTr,curSubWindows));
        aa(:,curKP) = a;
    end
        
end