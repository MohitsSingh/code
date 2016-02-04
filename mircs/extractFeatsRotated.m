function [feats,regions] = extractFeatsRotated(conf,currentID,partModels,regions,theta)

I = getImage(conf,currentID);
I = imrotate(I,theta,'bilinear','crop');
if (~exist('bowImages.mat','file'))
    regions = cellfun(@(x) imrotate(x,theta,'crop'),regions,'UniformOutput',false);
regions(~cellfun(@nnz,regions)) = [];
    bowImages = getBowFeats(I,regions);
    save bowImages bowImages regions
else
    load bowImages
end

% for k = 1:length(regions)
feats = getImageDescriptor([conf.featConf.bowmodel],regions,bowImages);


    function bowImages = getBowFeats(I,regions)
        bowImages = {};
        featConf = conf.featConf;
        for k = 1:length(featConf)
            [F,D] = vl_phow(I,featConf(k).featArgs{:});
            bins = minDists(single(D),single(featConf(k).bowmodel.vocab),5000);
            bowImages{k} = makeBowImage(I,F,bins);
        end
    end


end