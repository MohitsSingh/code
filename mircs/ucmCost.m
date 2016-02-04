function [ cost ] = ucmCost(I, ucm, x, y , T)
%UCMCOST Summary of this function goes here
%   Detailed explanation goes here

sz = size(ucm);
%T = [Tx Ty s];
% T(3) = 0;
x_ = x;
y_ = y;
[x,y] = deform_fn(sz,x,y,T);
inImage = inImageBounds(sz,[x y]);
x = x(inImage);
y = y(inImage);
cost_u = -mean(interp2(ucm,x,y,'bilinear'));
% inds = sub2ind2(size(ucm),[y(inImage) x(inImage)]);
% cost_u = -sum(ucm(inds));%/length(x);
% sig_ = .05;
% cost_d = exp(-(norm(T(1:3)).^2)/sig_);
%cost_d = exp(-(norm(T(1:2)).^2)/sig_);
cost_d = .1*norm(T(1:2)) + .3*abs(T(3));
lambda = .1;
cost = cost_u+lambda*cost_d;
if (toc > 1)
    clf; subplot(1,2,1);
    imagesc(I); axis image; hold on;
    plot(x_,y_,'r.');
    plot(x,y,'m+');
    subplot(1,2,2);
    imagesc(ucm); axis image; hold on;
    plot(x_,y_,'r.');
    plot(x,y,'m+');
    disp(cost);
    drawnow
    tic
end
end

