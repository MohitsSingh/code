%% initialize data...
%%
initStuff;

%% general plan:
% train keypoint predictor (without occlusion)
% train occlusion predictor
% train keypoint given occlusions predictor
% ok, now train with stronger features...
%% Create regression model for facial landmark locations.
X_train = getImageStackHOG(IsTr,64);
X_test = getImageStackHOG(IsT,64);
%%
phis_tr_p = normalize_coordinates(phisTr,IsTr);
kp_sel = 1; % left eye brow
xy_train = squeeze(phis_tr_p(:,kp_sel,1:2));
[phis_t_p,factors] = normalize_coordinates(phisT,IsT);
% (for each keypoint, train a local regressor too, to fix it's location
% a bit)
%%
% model = train_kp_regressor(X_train,xy);
models = train_kp_regressor_group((X_train),phis_tr_p(:,:,1:2));
% model_x = train(xy(:,1), sparse(double(X_train)), 'l1 -c 1', 'col');
%% visualize the first level regressions
figure(1); clf;
xy_pred = apply_kp_regressor((X_test),models);
xy_pred = reshape(xy_pred,2,29,[]);
z = zeros(1,1,length(factors));
z(:) = factors;
xy_pred = xy_pred.*repmat(z,2,29,1);
for p = 1:length(IsT)
    clf; imagesc2(IsT{p});
    plotPolygons(xy_pred(:,:,p)','m.');
    plotPolygons(squeeze(phisT(p,:,1:2)),'g.');
    drawnow
    %plotPolygons(factors(p)*xy_t(p,:),'g.');
    pause
end

%% learn a local regressor for each keypoint, given it's predicted neighborhood.
kp_sel = 1; % left eye brow
xy_train = squeeze(phisTr(:,kp_sel,1:2));
for p = 1:length(IsTr)
    clf; imagesc2(IsTr{p});
    m = squeeze(phisTr(p,:,1:2));
    %m = m(1,:);
    m = xy_train(p,:);
    %plotPolygons(squeeze(phisTr(p,:,1:2)),'g.');
    plotPolygons(m,'g.');
    drawnow
    %plotPolygons(factors(p)*xy_t(p,:),'g.');
    pause
end

%%


% 14, eye bottom
kp_sel = 17; % 17 left eye center

kp_sel = 25; % top lips top center
kp_sel = 26; % top lips bottom center
kp_sel = 27; % bottom lips top center
kp_sel = 24; % right lips corner
kp_sel = 23; % right lips corner

kp_sel = 25; % 17 left eye center

xy_train = squeeze(phisTr(:,kp_sel,1:2));
xy_test = squeeze(phisT(:,kp_sel,1:2));
sizes = cell2mat(cellfun2(@size2,IsTr));% height/width
nPatchesPerPoint = 10;
patchToFaceRatio = .5;
maxDeviation = .2;


gt_patches = {};
gt_devs = {};
for i = 1:nPatchesPerPoint
    i
    [kp_patches_train,deviations_train] = sampleLocalPatches(IsTr,xy_train,patchToFaceRatio,maxDeviation);
    gt_patches{i} = kp_patches_train;
    gt_devs{i} = deviations_train;
end

kp_patches_train = cat(2,gt_patches{:});
deviations_train = cat(1,gt_devs{:});

% sizes = cell2mat(cellfun2(@size2,kp_patches_train));% height/width
deviations_train_n = normalize_coordinates(deviations_train,kp_patches_train(:),true);
subWindowSize = 48;
X_train_local = getImageStackHOG(kp_patches_train,subWindowSize,true,false,8);
xy_test = squeeze(phisT(:,kp_sel,1:2));
[kp_patches_test,deviations_test,kp_offsets_test,new_centers_test] = sampleLocalPatches(IsT,xy_test,patchToFaceRatio,maxDeviation);
deviations_test_n = normalize_coordinates(deviations_test,kp_patches_test(:),true);
X_test_local = getImageStackHOG(kp_patches_test,subWindowSize,true,false,8);

local_model = train_kp_regressor(X_train_local,deviations_train_n);
% X_pred_local = apply_kp_regressor(X_test_local,local_model);
%%
for u = 1:1:length(kp_patches_test)
    
    curIm = IsT{u};
    
    curPatchCenter = new_centers_test(u,:);
    
    sz = size(curIm,1);
    prevCenter = new_centers_test(u,:);
    figure(1);
    
    jj = 20;
    
    diffs = zeros(100,1);
    for iIter = 1:100
        % computation
        
%         curBox,
        alpha_ = .01;
        [curCenter,curBox,curSubWindow,curX] = ...
            predict_next(prevCenter,curIm,patchToFaceRatio,subWindowSize,local_model,...
        alpha_);

% %         curBox = round(inflatebbox([prevCenter prevCenter],sz*patchToFaceRatio,'both',true));
% %         curSubWindow = cropper(curIm,curBox);
% %         curX = getImageStackHOG(curSubWindow,subWindowSize,true,false,8);
% %         curPredictedDeviation = sz*apply_kp_regressor(curX,local_model)';
% % %                 curPredictedDeviation = curPredictedDeviation/norm(curPredictedDeviation);
% %         alpha_ = .01;
% %         curCenter = prevCenter - alpha_*curPredictedDeviation;
%         curPredictedDeviation = X_pred_local(:,u)*size(curIm,1)';
        % visualization
        
        diffs(iIter) = norm(curCenter-xy_test(u,:));
        
        if (mod(iIter,jj)==0)
            
            clf;
            subplot(1,2,1);
            imagesc2(curIm);
            plotPolygons(curPatchCenter,'m*');
            plotPolygons(xy_test(u,:),'g*');
            plotPolygons(prevCenter,'rd','LineWidth',2);
            plotBoxes(curBox,'r--','LineWidth',2);
            quiver(curPatchCenter(1),curPatchCenter(2),-deviations_test(u,1),-deviations_test(u,2),0,'g-');
            quiver(prevCenter(1),prevCenter(2),-curPredictedDeviation(1),-curPredictedDeviation(2),0,'m-');
            subplot(1,2,2); 
            %imagesc2(kp_patches_test{u});            
            plot(diffs)
            %         plotPolygons(X_pred_local(u,:)*size(curIm,1),'r+');
            %         plotPolygons(X_pred_local);
            disp(norm(curCenter-xy_test(u,:)))
            drawnow
            
            
            pause(.1)
        end
        prevCenter = curCenter;
    end
end

% [phis_tr_p,factors_tr] = normalize_coordinates(phisTr,IsTr);

