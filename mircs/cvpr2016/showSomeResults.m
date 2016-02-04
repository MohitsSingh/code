function showSomeResults(res,outPath,fra_db,imdb,V,class_sel)
test_scores = res{V}.res_test(class_sel).curScores;
labels = [fra_db.classID];
isTrain = [fra_db.isTrain];
test_ids = find(~isTrain);
[r,ir] = sort(test_scores,'descend');
full_hires_path = '~/storage/fra_action_fcn_hires';
labels_full = {'none','face','hand','drink','smoke','bubbles','brush','phone'};
% ir = randperm(length(ir))
for it = 1:length(ir)
    it
    k = test_ids(ir(it));
    disp(['image index:' num2str(k)]);
    if labels(k)==class_sel,continue,end
    %     showPredsHelper(imdb,L,k);
    %     dpc
    p = j2m(outPath,fra_db(k));
    L = load(p);
    coarse_probs = L.scores_full_image;
    fine_probs = L.scores_hires;
    coarse_probs = bsxfun(@rdivide,exp(coarse_probs),sum(exp(coarse_probs),3));
    [~,coarse_pred] = max(coarse_probs,[],3);
    [~,fine_pred] = max(fine_probs,[],3);
    %     zoomBox = inflatebbox(region2Box(imdb.labels{k}>2),4,'both');
    zoomBox = inflatebbox(fra_db(k).faceBox,2.5,'both');
    %     showPredictions(single(imdb.images_data{k}),coarse_pred,coarse_probs,labels_full,1,zoomBox);
    %     showPredictions(single(imdb.images_data{k}),fine_pred,fine_probs,labels_full,2,zoomBox);
    
    %     p = j2m(full_hires_path,fra_db(k));
    %     L = load(p);
    %     fine_probs = L.scores_hires_full;
    %     fine_probs = bsxfun(@rdivide,exp(fine_probs),sum(exp(fine_probs),3));
    %     [~,fine_pred] = max(fine_probs(:,:,3));
    [h1_coarse,h2_coarse] = showPredictions(single(imdb.images_data{k}),coarse_pred,coarse_probs,labels_full,1,zoomBox);
    [h1_fine,h2_fine] = showPredictions(single(imdb.images_data{k}),fine_pred,fine_probs,labels_full,2,zoomBox);
    
    %     edit showPredsHelper
    
    ff ='figures/coarse_to_fine';
    %     saveas(h1_coarse,fullfile(ff,[fra_db(k).imageID(1:end-4) '_pred_coarse.png']));
    %     saveas(h2_coarse,fullfile(ff,[fra_db(k).imageID(1:end-4) '_prob_coarse.png']));
    %     saveas(h1_fine,fullfile(ff,[fra_db(k).imageID(1:end-4) '_pred_fine.png']));
    %     saveas(h2_fine,fullfile(ff,[fra_db(k).imageID(1:end-4) '_prob_fine.png']));
    dpc
end