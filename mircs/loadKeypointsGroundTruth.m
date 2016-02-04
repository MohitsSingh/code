function [kp_preds,goods] = loadKeypointsGroundTruth(imgData,reqKeyPoints)
        goods = true(size(reqKeyPoints));
        kp_preds = zeros(length(reqKeyPoints),3);
        %         if (imgData.isTrain)
        dbPath = '/home/amirro/storage/data/face_related_action';
        annoPath = j2m(dbPath,imgData);
        L = load(annoPath);
        curLandmarks = L.curLandmarks;
        kp_preds(:,end) = 1;
        for k = 1:length(reqKeyPoints)
            fn = reqKeyPoints{k};
            if (curLandmarks.(fn).skipped)
                goods(k) = false;
            else
                kp_preds(k,1:2) = curLandmarks.(fn).pts;
            end
        end
    end