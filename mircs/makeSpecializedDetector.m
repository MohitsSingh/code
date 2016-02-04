function [detector,q_train,q_test,conf,svm_scores] =makeSpecializedDetector(conf,subImages,suffix,override,inds,bundle)


% [detector,q_train,q_test_cup3,conf3] = makeSpecializedDetector(conf,cup3SubImages,'cup3',override,...
%     cupInds_3_abs,bundle);

train_imgs= bundle.train_imgs;
test_imgs= bundle.test_imgs;
t_train= bundle.t_train;
t_test= bundle.t_test;
train_imgs_large= bundle.train_imgs_large;
test_imgs_large= bundle.test_imgs_large;
roi_train= bundle.roi_train;
roi_test= bundle.roi_test ;
model= bundle.model;
descs_train = bundle.descs_train;
descs_test = bundle.descs_test;

if (isempty(strfind(suffix,'_ref')))
    curRects = selectSamples(conf,subImages,fullfile('specialized',[suffix 'Rects']));
else
    curRects = [];
end
% %inflate everything.
% return;
[detector,q_train,q_test,conf] = makeAndTest(conf,subImages,curRects,train_imgs,t_train,...
    test_imgs,t_test,suffix,override);
% close all;
% mmm = visualizeLocs2_new(conf,test_imgs,q_test.cluster_locs(1:100,:));
% mImage(mmm);

svm_scores = [];

% % % % % % q_train = arrangeDet(q_train,'index');
% % % % % % q_test = arrangeDet(q_test,'index');
% % % % % % 
% % % % % % test_rects_on_large = 32+roi_test/2;
% % % % % % b = q_test.cluster_locs(:,1:4);
% % % % % % [numRows] = BoxSize(test_rects_on_large);
% % % % % % res_test_rects = bsxfun(@times,b,numRows/100)+...
% % % % % %     [test_rects_on_large(:,1:2) test_rects_on_large(:,1:2)];
% % % % % % 
% % % % % % train_rects_on_large = 32+roi_train/2;
% % % % % % b = q_train.cluster_locs(:,1:4);
% % % % % % [numRows] = BoxSize(train_rects_on_large);
% % % % % % res_train_rects = bsxfun(@times,b,numRows/100)+...
% % % % % %     [train_rects_on_large(:,1:2) train_rects_on_large(:,1:2)];
% % % % % % 
% % % % % % feats_train = getBOWFeatures(conf,model,train_imgs_large,round(res_train_rects),descs_train);
% % % % % % feats_test = getBOWFeatures(conf,model,test_imgs_large,round(res_test_rects),descs_test);
% % % % % % % return;
% % % % % % % m_train = multiCrop(conf,train_imgs,round(q_train.cluster_locs),[80 80]);
% % % % % % % m_test= multiCrop(conf,test_imgs,round(q_test.cluster_locs),[80 80]);
% % % % % % % f_not = find(~t_train);
% % % % % % % f_not = f_not(1:4:end);
% % % % % % [w,~,~,~,svm_model] = train_classifier(feats_train(:,inds),feats_train(:,~t_train),[],[],0);
% % % % % % [~,~,svm_scores] = svmpredict(zeros(size(feats_test,2),1),feats_test',svm_model);
