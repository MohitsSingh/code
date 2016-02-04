function [matches, negs] = evaluateModel(...
    testImages, testBoxes, testBoxImages, w, hogCellSize, scales, debugging)

clear matches ;
negs = {} ;
for i=1:numel(testImages)
    % Detect on test image
    if (ischar(testImages{i}))
        im = imread(testImages{i}) ;
    else
        im = testImages{i};
    end
    im = im2single(im) ;
    [detections, scores, hog] = detect(im, w, hogCellSize, scales) ;
    
    % Non-maxima suppression
    keep = boxsuppress(detections, scores, 0.5) ;
    keep = find(keep) ;
    keep = vl_colsubset(keep, 15, 'beginning') ;
    detections = detections(:, keep) ;
    scores = scores(keep) ;
    
    % Find all the objects in the target image
    ok = find(strcmp(testImages{i}, testBoxImages)) ;
    gtBoxes = testBoxes(:, ok) ;
    gtDifficult = false(1, numel(ok)) ;
    matches(i) = evalDetections(...
        gtBoxes, gtDifficult, ...
        detections, scores) ;
    
    if (debugging)
        % Visualize progres
        clf;
        subplot(1,3,[1 2]) ;
        imagesc(im) ; axis equal ; hold on ;
        labels = arrayfun(@(x)sprintf('%d',x),1:size(detections,2),'uniformoutput',0) ;
        sp = fliplr(find(matches(i).detBoxFlags == -1)) ;
        sn = fliplr(find(matches(i).detBoxFlags == +1)) ;
        vl_plotbox(detections(:, sp), 'r', 'linewidth', 1, 'label', labels(sp)) ;
        vl_plotbox(detections(:, sn), 'g', 'linewidth', 2, 'label', labels(sn)) ;
        vl_plotbox(gtBoxes, 'b', 'linewidth', 1) ;
        title(sprintf('Image %d of %d', i, numel(testImages))) ;
        axis off ;
        
        subplot(1,3,3) ;
        vl_pr([matches.labels], [matches.scores]) ;
    end
    % If required, collect top negative features
    if nargout > 1
        overlaps = boxoverlap(gtBoxes, detections) ;
        overlaps(end+1,:) = 0 ;
        overlaps = max(overlaps,[],1) ;
        detections(:, overlaps >= 0.25) = [] ;
        detections = vl_colsubset(detections, 10, 'beginning') ;
        negs{end+1} = extract(hog, hogCellSize, scales, w, detections) ;
    end
    
    % Break here with the debugger
    drawnow ;
end

if nargout > 1
    negs = cat(4, negs{:}) ;
end