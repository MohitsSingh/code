function [d] = phowColorDesc(m)
%TINYDESC Summary of this function goes here
%   Detailed explanation goes here
%     m = reshape(m,[],3);

%         m = imresize(m,.5);
%         d = m(:);

[~, d] = vl_phow(m, 'Verbose',false, 'Sizes', [1 2 3 4], 'Step', 2);
% [~,d] = vl_phow(m,'Sizes',2,'Step',6,'Color','HSV');
% d = d(:);
% figure,imshow(m);
%     vl_plotframe(f([1 2 4 3],:));
% vl_plotsiftdescriptor(d(:,1),f([1 2 4 3],1));
%     d = mean(m);

end

