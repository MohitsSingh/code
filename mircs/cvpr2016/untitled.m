prevEnt = ent(scores1);
img_orig = single(I1);
curImage = single(img_orig);
%%
N  =0;
% curImage = curImage+rand(size(curImage))*50;
ENTROPY = 1;
OTHER_OBJECT = 2;
target_object = 986; % daisy :-)
%target_object = 286;
target_object = 1;
% target_object = 1:100:1000;
goalType = OTHER_OBJECT;
if goalType==ENTROPY
    prevScore = prevEnt;
else
    z_target = zeros(size(scores,1),1);
    z_target(target_object) = 1;
%     z_target = z_target/norm(z_target);
%     prevScore = l2(normalize_vec(z_target)',scores1');
    prevScore = -z_target'*scores1;
end
%%
% prevEnt = inf;
% goalType = ENTROPY;
% if goalType==ENTROPY
%     prevScore = prevEnt;
% else
%     z_target = zeros(size(scores,1),1);
%     z_target(target_object) = 1;
%     prevScore = -z_target'*scores1;
% end

for t = 1:1000
    t
    % do it in batches....
    batchSize = 128;
    z = randn([size(curImage) batchSize]);
    I4 = bsxfun(@plus,curImage,z);
    [curScores,bestScore,best] = predictAndShow(I4,net,0);
    if goalType==ENTROPY
        scoreFunction = ent(curScores);
        [curScore,iv] = min(scoreFunction);
    else
%         R = l2(normalize_vec(z_target)',normalize_vec(curScores)');
%         [curScore,iv] = min(R);
        [curScore,iv] = max(z_target'*curScores);
        curScore = -curScore;
    end
    if curScore < prevScore
        prevScore = curScore
        curImage = squeeze(I4(:,:,:,iv));
%         curImage = curImage-min(curImage(:));
%         curImage = 255*(curImage/max(curImage(:)));
        N = N+1;
        if (mod(N,1)==0)
            curScore
            figure(4) ; clf ;
            subplot(1,3,1);imagesc2(curImage/255) ;title('current');
            title(sprintf('%s\n (%d)\n score %.3f',...
                net.classes.description{best(iv)}, best(iv), bestScore(iv))) ;
            subplot(1,3,3); bar(curScores(:,iv));
            subplot(1,3,2);imagesc2(img_orig/255) ; title('orig');
            dpc(.1)
        end
    end
    
end
