

%%
for k =1:length(lipImages_test_2)
    if (~t_test_tt(k))
        continue;
    end
    subplot(2,2,1);
    imagesc(lipImages_test_2{k}); axis image;
    subplot(2,2,2);
    imagesc(test_gbps(k).gPb_thin);axis image;
    ucm = contours2ucm(test_gbps(k).gPb_orient);
    subplot(2,2,3);
    imagesc(ucm);axis image;
    subplot(2,2,4);
     kk = 0.2; %64; %100
    bdry = (ucm >= kk);
    imagesc(bwlabel(ucm <= kk)); axis image;
    %         break
    pause;
%     train_face_labels{q} = bwlabel(ucm <= k);

end

%%

train_gpbs = struct;
for k =1:length(train_faces_tt)
    k
    [train_gbps(k).gPb_orient, train_gbps(k).gPb_thin, train_gbps(k).textons] = globalPb(lipImages_train_tt{k});        
end

save gpbs.mat test_gpbs train_gbps;