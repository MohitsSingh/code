function bowImage = makeBowImage(I,F,bins)
scales = unique(F(4,:));
bowImage = zeros([dsize(I,1:2),length(scales)]);
for iScale = 1:length(scales)
    curFrames = F(:,F(4,:)==scales(iScale));
    subs_ = [curFrames(2,:)',curFrames(1,:)',ones(size(curFrames,2),1)*iScale];
    bowImage(sub2ind2(size(bowImage),subs_)) =...
        bins(F(4,:)==scales(iScale));
end