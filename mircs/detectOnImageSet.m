function all_rects = detectOnImageSet(classifier,imgDir,recs,prms)
all_rects = {};
for iImg = 1:length(recs)
    progress(iImg,length(recs));
    I = imread(fullfile(imgDir,[recs(iImg).filename '.JPEG']));
    I = im2double(I);
    %         all_bbs = detect_rotated(I,curClassifier,cell_size,features,detection,threshold,dTheta,true);
    %         I = max(0,min(1,I+rand(size(I))*.1));
    %         I = padarray(I,[50 50],0,'both');
    rects = detect(I,classifier.weights,classifier.bias,classifier.object_sz,...
        prms.cell_size,prms.features,prms.detection,prms.threshold);
    if (isempty(rects))
        continue;
    end
    rects(:,3:4) = rects(:,3:4) + rects(:,1:2);
    rects = clip_to_image(rects,I);
    
    all_rects{end+1} = [rects,ones(size(rects,1),1)*iImg];
    
%             A = computeHeatMap(I,rects,'max');
    
%             sc(cat(3,A,I),'prob'); pause    
end
all_rects = cat(1,all_rects{:});
