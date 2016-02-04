% gistFeaturesPath = 'gistFeatures.mat';
% if (exist(gistFeaturesPath,'file'))
% %     load(gistFeaturesPath);
% else
function gFeatures = getGists(VOCopts,train_ids,gistsPath)
gFeatures =  zeros(960,length(train_ids),'single');
for iID = 1:length(train_ids)
    
    currentID = train_ids{iID};
    if (mod(iID,50)==0)
        iID
    end
    gistPath = fullfile(gistsPath,[num2str(currentID) '.mat']);
    if (~exist(gistPath,'file'))
        I = readImage(VOCopts,currentID);
        g =GlobalDescriptor(I);
        save(gistPath,'g');
    else
        load(gistPath);
        
    end
    gFeatures(:,iID) = g;
end