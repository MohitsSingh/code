%% now for the profile....
nReqKeypoints = 39; % for profile, require 39
scores = [all_lm.s];
nKP = arrayfun(@(x) size(x.xy,1),all_lm);
cur_lm = all_lm(scores >= min_score & nKP==nReqKeypoints);
min_score = .2;
cur_inds = all_inds(nKP==nReqKeypoints & scores >= min_score);
ss_1 = [cur_lm.s];
[r,ir] = sort(ss_1,'descend');
faces_and_landmarks = struct('I',{},'xy',{},'c',{});
n = 0;
for u = length(ir):-1:1 % subsample...
    n = n+1;
    %length(ir):-50:1
    k = ir(u);
    num2str([u r(u)])
    curInd = str2num(d(cur_inds(k)).name(1:end-4));
    I = ims{curInd};
    U = imrotate(I,cur_lm(k).rotation,'bilinear','crop');
    
    faces_and_landmarks(n).I = U;
    faces_and_landmarks(n).bbox = round(inflatebbox([1 1 fliplr(size2(U))],1/1.3,'both',false));
    faces_and_landmarks(n).xy = boxCenters(cur_lm(k).xy);
    faces_and_landmarks(n).c = cur_lm(k).c;
    wasFlipped = false;
    if (cur_lm(k).c > 10)
                    disp('flipping');
                    wasFlipped  = true;
        faces_and_landmarks(n).I = flip_image(U);
        %faces_and_landmarks(n).bbox = flip_box(faces_and_landmarks(n).bbox,U);
        faces_and_landmarks(n).xy(:,1) = size(U,2)-faces_and_landmarks(n).xy(:,1);
    else
        %             disp('not flipping');
    end
    if (wasFlipped )        
    clf;
    imagesc2(faces_and_landmarks(n).I);
    plotPolygons(faces_and_landmarks(n).xy,'r+');
    plotBoxes(faces_and_landmarks(n).bbox);
    drawnow;pause(.01)
    end
end