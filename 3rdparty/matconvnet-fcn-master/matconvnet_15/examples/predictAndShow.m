function [scores2,bestScore,best] = predictAndShow(I2,net,fignum)




im_ = normalize_image(I2,net);

% run the CNN
res = vl_simplenn(net, gpuArray(im_)) ;
% show the classification result
scores2 = squeeze(gather(res(end).x)) ;
[z,iz] = sort(scores2,'descend');
% iz(1:5)'
% row(net.classes.description(iz(1:5)))
[bestScore, best] = max(scores2,[],1) ;
if fignum > 0
    figure(fignum) ; clf ;
    subplot(1,2,1);imagesc2(I2) ;
    title(sprintf('%s (%d), score %.3f',...
        net.classes.description{best}, best, bestScore)) ;
    subplot(1,2,2); bar(scores2);
end