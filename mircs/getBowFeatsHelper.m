function bowFeats = getBowFeatsHelper(conf,currentID,masks,featConf)
if (ischar(currentID))
    [~,name,~] = fileparts(currentID);
    bowImages = {};
    for k = 1:length(featConf)
        bowFile = fullfile(conf.bowDir,[name '_' featConf(k).suffix '.mat']);
        if (exist(bowFile,'file'))
            %     load(bowFile,'F','bins');
            load(bowFile)
        else
            conf.get_full_image = true;
            I = getImage(conf,currentID);
            [F,D] = vl_phow(I,featConf(k).featArgs{:});
            bins = minDists(single(D),single(featConf(k).bowmodel.vocab),5000);
            bowImage = makeBowImage(I,F,bins);
        end
        bowImages{k} = bowImage;
    end
else
    for k = 1:length(featConf)
        I = im2single(currentID);
        [F,D] = vl_phow(I,featConf(k).featArgs{:});
        bins = minDists(single(D),single(featConf(k).bowmodel.vocab),5000);
        bowImage = makeBowImage(I,F,bins);
        bowImages{k} = bowImage;
    end
end

%feats = struct('bowImage',{bowImages});

% bowFeats = getBOWFeatures(conf,[featConf.bowmodel],{currentID},masks,bowImages);

bowFeats = getImageDescriptor([featConf.bowmodel],masks,bowImages);
end
