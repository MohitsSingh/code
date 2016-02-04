function [R,I] = doMaskSearch_deep(I,net_deep,layers,w,cutoff)
imRect = [1 1 fliplr(size2(I))];
curRect = imRect;
% curRect = makeSquare(imRect,true);
sz = size2(I);
nIters = 0;
R = zeros(size2(I));
maxIters =15;
scores = -inf(1,maxIters);
I = imResample(I,256/size(I,1),'bilinear');
% segments = vl_slic(im2single(vl_xyz2lab(vl_rgb2xyz(I))), 10, 0.1);
% segments = vl_slic(im2single(((I))), 50, 0.1);
% [I,c] = paintSeg(I,segments);
imo = prepareForDNN({I},net_deep,false);
% run through the network once
dnn_res_head = vl_simplenn(net_deep, imo);
% only need the to run the next layer, which is fully connected
if nargin < 5
    cutoff = 15; % one before fully connected layer (original)
end
% cutoff = 13; % result of 5th convolutional layer
x =  dnn_res_head(cutoff).x;


SINGLE_CELL = 1;
BINARY_SEARCH = 1;
search_method = SINGLE_CELL;

net_tail = net_deep;
net_tail.layers = net_tail.layers(cutoff:end);
sz_x = size2(x);
R = zeros(sz_x);

switch search_method
    case SINGLE_CELL
        
        %%xx = zeros([size(x) prod(sz_x)]);
        
        
        
        xx = repmat(x,[1 1 1 prod(sz_x)]);
        tic
        n = 0;
        for jj = 1:sz_x(2)
            for ii = 1:sz_x(1)
                n = n+1;
                xx(ii,jj,:,n) = 0;
                %         dnn_res_tail = vl_simplenn(net_tail,x_1);
                %         R(ii,jj) = w*dnn_res_tail(2).x(:);
            end
        end
        dnn_res_tail = vl_simplenn(net_tail,xx);
        
        R = reshape(w*squeeze(dnn_res_tail(end).x),sz_x);
    case BINARY_SEARCH
        grid_size = sz_x;
        
        
end
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
