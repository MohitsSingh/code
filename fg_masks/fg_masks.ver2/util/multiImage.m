function [I] = multiImage(images,sz);
%MULTIIMAGE Summary of this function goes here
%   Detailed explanation goes here
p = round(length(images).^.5);
I = {};
q = 0;
for ii = 1:p
    cur_row = {};
    for jj = 1:p
        q = q+1;
        if (q <= length(images))
            cur_row{jj} = imresize(images{q},sz);
        else
            cur_row{jj} = zeros([sz 3]);
        end
    end
    I{ii} = cat(2,cur_row{:});    
end
I = cat(1,I{:});