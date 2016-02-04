function [R,I] = doMaskSearch_deep2(I,net_deep,layers,w,cutoff)
imRect = [1 1 fliplr(size2(I))];
curRect = imRect;
% curRect = makeSquare(imRect,true);
sz = size2(I);
nIters = 0;
R = zeros(size2(I));
maxIters =15;
scores = -inf(1,maxIters);
%I = imResample(I,256/size(I,1),'bilinear');
% segments = vl_slic(im2single(vl_xyz2lab(vl_rgb2xyz(I))), 10, 0.1);
% segments = vl_slic(im2single(((I))), 50, 0.1);
% [I,c] = paintSeg(I,segments);
imo = prepareForDNN({I},net_deep,false);
% run through the network once
dnn_res = vl_simplenn(net_deep, imo);
curLayer = dnn_res(length(dnn_res));
vals = squeeze(dnn_res(end).x).*w(:);
iLayer = 14;
while(iLayer > 0)
    [v,iv] = sort(vals);
    bestUnit = iv(1);
    net_deep.layers{iLayer}
end


% backwards: find the important inner units contributing to the
% classification

R = reshape(w*squeeze(dnn_res_tail(end).x),sz_x);
% toc
%
% tic
% for ii = 1:sz_x(1)
%     for jj = 1:sz_x(2)
%         x_1 = x;
%         x_1(ii,jj,:) = 0;
%         dnn_res_tail = vl_simplenn(net_tail,x_1);
%         R(ii,jj) = w*dnn_res_tail(2).x(:);
%     end
% end
% toc

R = imResample(exp(-R),size2(I),'nearest');
%imagesc(dnn_res(15).x(:,:,3))

% do a depth first search?


end
