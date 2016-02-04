ids = train_ids(train_labels);
ids = test_ids(test_labels);
ids = test_ids;
% % 
% % image_inds = zeros(size(ids));
% % rects = zeros(length(ids),6,4);
% % 
pp = pwd;
rects = {};
for iImage = 1:length(ids)
%     I = getImage(conf,ids{iImage});
% %     I = imrotate(I,30,'bilinear');
%     cd /home/amirro/code/3rdparty/voc-release5
%     [ds, bs] = process(I,partModelsDPM{2});
%     cd ~/code/mircs
%     imshow(I); hold on; plotBoxes2(ds(1:5,[2 1 4 3]));
    %load(fullfile('~/storage/dpm_s40',strrep(ids{iImage},'.jpg','.mat')));
%     for iModel = 1:4
%         rects{iModel,iImage} = modelResults(iModel).ds;
%     end
end
% % 
% % for iImage = 1:length(ids)
% %     iImage
% %     load(fullfile('~/storage/dpm_s40',strrep(ids{iImage},'.jpg','.mat')));
% %     for iModel = 1:4
% %         rects(iImage,:,iModel) = modelResults(iModel).ds(1,:);
% %     end
% % end
% % modelNames = {'cup','hand','straw','bottle'};
% % for iModel = 1:4
% %     [s,is] = sort(rects(:,6,iModel),'descend');
% %     for jj = 1:10 % show top ten results...
% %         I = getImage(conf,ids{is(jj)});
% %         clf; imagesc(I); hold on; axis image;
% %         plotBoxes2(rects(is(jj),:,[2 1 4 3]),'color','r','LineWidth',2);
% %         title(modelNames{iModel});
% %         pause;
% %     end
% % end
% % 
% % ids = ids(randperm(length(ids)));
% % for k = 1:length(ids)
% %     I = getImage(conf,ids{k});
% %     load(fullfile('~/storage/dpm_s40',strrep(ids{k},'.jpg','.mat')));
% %     close all;
% %     for iModel = 1:length(modelResults)
% %         clf; imshow(I); hold on;
% %         ds  =modelResults(iModel).ds;
% %         ds = ds(1:min(5,size(ds,1)),:);
% %         plotBoxes2(ds(:,[2 1 4 3]),'color','r','LineWidth',2);
% %         curClass = strrep(modelResults(iModel).class,'_',' ');
% %         title([curClass ' : ' num2str(modelResults(iModel).ds(1,end))]);
% %         
% %         pause;
% %     end
% % end
% % 

ids = test_ids;
% ids = test_ids(test_labels);
imgScores = -inf*ones(size(ids));

for iImage = 1:length(ids)
    if (mod(iImage,100)==0)
        iImage
    end
    
    try 
    
        load(fullfile('~/storage/seq_res_s40',strrep(ids{iImage},'.jpg','.mat')),'res');
    catch e
        e
    end
%     res.scores
    imgScores(iImage) = res.scores(1);    
end

[s,is] = sort(imgScores,'descend');
conf.get_full_image = true;
for iImage = 13:length(ids)
    I = getImage(conf,ids{is(iImage)});
    clf; subplot(1,2,1);imagesc(I);axis image
    title(num2str(s(iImage)));
    load(fullfile('~/storage/seq_res_s40',strrep(ids{is(iImage)},'.jpg','.mat')),'res');
    regions = getRegions(conf,ids{is(iImage)},false);
    Z = zeros(dsize(I,1:2));
    for r = 1:length(res.allRegions{1})-1
        Z = Z+regions{res.allRegions{1}(r)};
    end
%     Z = Z/3;
    subplot(1,2,2); imagesc(repmat(Z,[1 1 3]).*I); axis image;
    title(res.parts{1});
    pause;   
end

[prec,rec,aps] = calc_aps2(imgScores,test_labels);
plot(rec,prec)


