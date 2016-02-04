function [cross_val_results,classifiers] = get_cross_val_classification(feats,labels1,ovps,nFolds,stage_params,classes);

cross_val_results = cell(length(classes),length(labels1));

fold_sel = 1+mod(0:length(feats)-1,5);
for iFold = 1:nFolds
    curVal = fold_sel==iFold;
    curTrain = ~curVal;
    trainFeats = cat(2,feats{curTrain});
    trainLabels = cat(2,labels1{curTrain});
    trainOvps = cat(2,ovps{curTrain});
    for iClass = 1:length(classes)
        classifiers{iFold,iClass} = train_region_classifier(trainFeats,trainLabels,classes(iClass),stage_params);
        for u = 1:length(curVal)
            u
            if (curVal(u))
                cross_val_results{iClass,u}=apply_region_classifier(classifiers{iFold,iClass},feats{u},stage_params);
            end
        end
    end
    %     valFeats = cat(2,feats{curTrain});
    %     curLabels = cat(2,labels1{curTrain});
end