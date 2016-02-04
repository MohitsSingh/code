%% Experiment 0019 %%
% Create specialized functions for detecting different elements of
% drinking.
if (~exist('conf','var'))
    initpath;
    config;
    load ~/storage/misc/imageData_new;
    resPath = '~/mircs/experiments/experiment_0019/';
    mkdir(resPath);
    addpath('/home/amirro/code/3rdparty/elsd_1.0/');
    addpath('/home/amirro/code/3rdparty/datastrcuture/');
    addpath('/home/amirro/code/3rdparty/hash/');
    addpath('/home/amirro/code/3rdparty/DT');
    %addpath(genpath('/home/amirro/code/3rdparty/markSchmidt/UGM'));
    elsd_output_dir  ='~/storage/s40_elsd_output/';
end

addpath(genpath('~/code/3rdparty/geom2d'));
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
groundTruth = consolidateGT(conf,'train',false);
[gt_train_g,keys] = groupBy(groundTruth,'sourceImage');
%
% conf.get_full_image = true;
% for k = 1:length(gt_train)
%     curID=  gt_train_g(k).key;
%     I = getImage(conf,curID);
%     curGroup = gt_train_g(k).group;
%     for kk = 1:length(curGroup)
%         if (strcmp(curGroup(kk).name,'mouth'))
%             clf; imagesc(I); axis image;hold on;
%             plot(curGroup(kk).polygon.x,curGroup(kk).polygon.y,'r+');
%             pause;
%         end
%     end
% end

% define a set of small geometric configuration from which to start.
%% start with cups - again.

%% start with cups.
% [groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
% [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
gtParts = {groundTruth.name};
imageIDS = {groundTruth.sourceImage};
isObj = cellfun(@any,strfind(gtParts,'cup'));

%% iterate over true train images.
conf.get_full_image = true;
% for test : 19, 22, 47, 66, 126, 149

% interesting straw cases: 57, 60. 63 (straw) now complete the NMS, (but dont remove pairs
% of parallel lines) and the bottom-of-cup test
% for bottles, need "side" ellipses or another feature: NOTE
% bottles essentially same as cups,but more horizontal there is a *bottom* ellipse but the top
% one is harder to find.

% **bottles**
% not for bottles of small resolution the bottom line is not necessarily
% found as an ellipse. So find a U shape pointing to the face.
% bottles: 6- low res.
% 12 - ellipses found. complete by: 1. lines connecting them 2.
% non-accidental segment.
% 18 - again, bad resolution, features missed
% 29 - interesting, found as cup.
% 33 - bottom ellipse found
% 39 -bottom ellipse found. top ellipse found, but just by luck.
% 40 - both ellipses found.
% 48: bottom partially occluded by hand. Add the hand detection as a
% skipping (gap) node! Also, there is a straw; when a straw is found, look
% for a bottle too (e.g, advance to multiple states)
% 50 : top and bottom found. what to do when ellipse is too "full" (i.e,
% the most part of the ellipse is found, and then the secant doesn't define
% the orientation well, but the major axis does. As a rule, if
% theta2-theta1 is small enough, use the secant; otherwise, use the major
% axis (but the major axis will make the "middle vector" ambiguous.
% 53: we can see that in most cases the orientation of the bottom ellipse
% is more reliable (because it is larger and probably less occluded)
%*Generally, maybe use only one line for the straw detection?

% 62 - not actually a straw but a stream of water
% 67 - bottom ellipse not found, because shape of bottom is
% not elliptical...
% 75,76,80
% 81: observation: I think the hand is important to determine the held object.

% 85: tip of bottle hardly visible (most transparent) and there is no
% detected ellipse at the bottom. But there is a hand. However, the
% bottle+face are detected as a connected segment wherein the shape of the
% bottle is delineated well (except it is not separated from the hand
% holding it). But a U shape, pointing toward the face, is definitely there
% on the segment, so I can use that too.
%

%88 : bottle touches edge of image, but segment containing it is well
%behaved, shapewise.
from_gt =false;
%for k = 86:length(gt_train_g) % 66 ,74
for k = 1:length(train_ids)
    % for k = 12
    k
    %     currentID = 'drinking_168.jpg';
    if (from_gt)
        currentID = gt_train_g(k).key;
    else
        currentID = train_ids{k};
        if (imageData.train.faceScores(k) < -.6)
            continue;
        end
        conf.get_full_image = false;
    end
    
    if (isempty(strfind(currentID,'drink'))), continue, end;
    
    [I,I_rect] = getImage(conf,currentID);
    %     imshow(I);
    %     pause;continue;
    close all;
    [ucm,gpb_thin] = loadUCM(conf,currentID);
    %        figure,imagesc(gpb_thin);axis image; hold on;
    %      plot_svg(lines_,ellipses_);
    %     figure,imagesc(ucm);
    img = 1*(1-gpb_thin);
    img(gpb_thin<=.1) = 10;
    [D R] = DT(img);
    %     figure,imagesc(D)
    %     shortestPath(D,I);
    %     pause;
    %     continue
    if (from_gt)
        
        curGroup = gt_train_g(k).group;
        mouthPoint = [];
        facePoly = [];
        t = 0;
        for kk = 1:length(curGroup)
            if (strcmp(curGroup(kk).name,'mouth'))
                t = t+1;
                mouthPoint = [curGroup(kk).polygon.x,curGroup(kk).polygon.y];
            end
            if (strcmp(curGroup(kk).name,'face'))
                t = t+1;
                facePoly = [curGroup(kk).polygon.x,curGroup(kk).polygon.y];
            end
            
            if (t==2)
                break;
            end
        end
        
        %     if (isempty(strfind([curGroup.name],'bottle'))), continue, end;
        
        clf;imagesc(I); axis image; hold on;
        plot(facePoly(:,1),facePoly(:,2),'g-');
        plot(mouthPoint(1),mouthPoint(2),'r+');
    end
    elsd_file = fullfile(elsd_output_dir,strrep(currentID,'.jpg','.txt'));
    A = dlmread(elsd_file);
    [I] = getImage(conf,currentID);
    
    [lines_,ellipses_] = parse_svg(A,I_rect);
    
    resPath = fullfile('~/storage/s40_elsd_gpb',strrep(currentID,'.jpg','.mat'));
    
    if (~exist(resPath,'file')),
        disp(['skipping ' currentID]);
        continue
    end
    L = load(resPath);
    I = getImage(conf,currentID);
    [lines_1,ellipses_1] = parse_svg(L.A,I_rect);
    
    %     lines_ = [];
    %     ellipses_ = [];
    
    lines_ = [lines_;lines_1];
    ellipses_ = [ellipses_;ellipses_1];
    %
    %     pick_lines = lineNMS(lines_);
    %     lines_ = lines_(pick_lines,:);
    % %
    %     pick_ellipses = ellipseNMS(ellipses_);
    
    %     pick_ellipses_ = ellipse_NMS(lines_);
    %[ output_args ] = shortestPath(energyFun,currentID)
    
    
    %saveas(gcf,fullfile('~/notes/images/state_machine/',strrep(currentID,'.jpg','.fig')));
    
    clf; figure;imagesc(I); axis image; hold on;
    %saveas(gcf,fullfile('~/notes/images/state_machine/',strrep(currentID,'.jpg','_elsd.fig')));
    %         plot_svg(lines_,ellipses_);
    %         pause;
    
    %     primitives = getKASPrimitives(conf,imageID);
    
    %clf; figure;imagesc(I); axis image; hold on; follow_geometry_2(conf,I,lines_,ellipses_,imageData.train,currentID,facePoly,mouthPoint);
    %     I = gpb_thin;
    if (from_gt)
        clf; figure;imagesc(I); axis image; hold on; follow_geometry_2(conf,I,lines_,ellipses_,imageData.train,currentID,facePoly,mouthPoint);
    else
        
        clf; figure;imagesc(I); axis image; hold on; follow_geometry_2(conf,I,lines_,ellipses_,imageData.train,currentID);
    end
    disp('finished');
    %     follow_geometry(I,lines_,ellipses_,imageData.train,currentID,facePoly,mouthPoint);
    pause
end

