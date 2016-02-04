function featPath = getFeatPath( conf,imageID )
%getfeatpath summary of this function goes here
%   detailed explanation goes here
[~,name,~] = fileparts(imageID);
featPath = fullfile(conf.featPath,[name '.mat']);

end

