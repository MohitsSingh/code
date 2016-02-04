function [sets1,sets2] = splitSets( sets,dim)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
sets1 = {};
sets2 = {};
for t = 1:length(sets)
    t
    if dim == 4
        sets1{t} = sets{t}(:,:,:,1:2:end);
        sets2{t} = sets{t}(:,:,:,2:2:end);
    elseif dim == 3
        sets1{t} = sets{t}(:,:,1:2:end);
        sets2{t} = sets{t}(:,:,2:2:end);
    elseif dim == 2
        sets1{t} = sets{t}(:,1:2:end);
        sets2{t} = sets{t}(:,2:2:end);
    else
        sets1{t} = sets{t}(1:2:end);
        sets2{t} = sets{t}(2:2:end);
    end
end



end

