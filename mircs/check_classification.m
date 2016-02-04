initpath;
config;
conf.get_full_image = true;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');


test_results = {};
test_inds = {};
region_ids ={};
%%
for k = 1:length(test_ids)
    k
    if (length(test_results)<k || isempty( test_results{k}))
        
        currentID = test_ids{k};
%         currentID = 'walking_the_dog_197.jpg';
        
        curResultFile = strrep(fullfile('~/storage/res_s40',currentID),'.jpg','.mat');
        load(curResultFile);       
        nans = isnan(regionScores);
        regionScores(nans) = -1000;
        regions_ids{k} = 1:length(regionScores);
        test_results{k} = regionScores;
        test_inds{k} = k*ones(size(regionScores));
    end
end
 
% save classification_data.mat test_results test_inds

%%

% 
% means = cellfun(@mean,test_results);
% figure,plot(means)
% 

rrr = cat(2,test_results{:});
iii = cat(2,test_inds{:});
ggg = cat(2,regions_ids{:});
[r,ir] = sort(rrr,'descend');
%%
visitedImages = false(size(test_ids));

for k = 1:length(ir)
    imageIndex = iii(ir(k));
    if (~test_labels(imageIndex))
        continue;
    end
    if (visitedImages(imageIndex))
        continue;
    end
    visitedImages(imageIndex) = true;
    curImage = getImage(conf,test_ids{imageIndex});
    curRegionsFile = strrep(fullfile('~/storage/gpb_s40',test_ids{imageIndex}),'.jpg','_regions.mat');
    load(curRegionsFile)    
    clf;
    subplot(1,2,1);imagesc(curImage);axis image;
    curRegion = regions{ggg(ir(k))};
    subplot(1,2,2);imagesc(curImage.*repmat(curRegion,[ 1 1 3]));axis image;
    pause
end
