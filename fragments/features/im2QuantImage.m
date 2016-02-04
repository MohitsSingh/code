function [quantImage] = im2QuantImage( F,quantized,sz,globalOpts)
%IM2QUANTIMAGE Creates visual word image from a single scale...
%   Detailed explanation goes here
fff = F(4,:);
fff = fff==globalOpts.scale_choice;
F = F(:,fff);
quantized = quantized(fff);
quantImage = accumarray([F(2,:)',F(1,:)'],quantized,sz);
quantImage(quantImage == 0) =  globalOpts.numWords+1;
end

