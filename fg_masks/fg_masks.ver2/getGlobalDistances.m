function [IB,IG,dict] = getGlobalDistances(conf,trainSet,testSet)
% return distance matrices for GIST and BOW between images from testSet to
% images in trainSet
% calculates a BOW dictionary as well if necessary and returns it;
% create a dictionary for BOW features

dict = learnDictionary(conf.VOCopts,trainSet);

[gFeaturesTrain,bowHistsTrain] = getFeatures(trainSet);
[gFeaturesTest,bowHistsTest] = getFeatures(testSet);

% now we can calculate distances in BOW and GIST forms.
D_bow = l2(bowHistsTest',bowHistsTrain');
% D_bow = squareform(pdist(bowHists'));
D_gist = l2(gFeaturesTest',gFeaturesTrain');

[~,IB] = sort(D_bow,'ascend');
[~,IG] = sort(D_gist,'ascend');
% no need for the features, only the distances are of interest
% clear D_gist D_bow gFeatures bowHists;


    function [gFeatures,bowHists] = getFeatures(imageSet)
        % retrieve / compute gist features
        gFeatures = getGists(conf.VOCopts,imageSet,'data/gists');
        
        % retrieve / calculate bow features
        bowHists = getBowHists(conf,imageSet,dict);
        
        % experimental - I thing this slightly improves results.
        bowHists = vl_homkermap(bowHists,1,'KChi2','gamma',.7);
        gFeatures = vl_homkermap(gFeatures,1,'KChi2','gamma',.7);
        
        
    end
end