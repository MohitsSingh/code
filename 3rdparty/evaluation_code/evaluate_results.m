function ap = evaluate_results(boxes)

% ap = evaluate_results(boxes)
% Score bounding boxes using the PASCAL development kit.

cls = 'hand';
testset = 'hand_test_big';
globals;
init;
year = VOCyear;
ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% write out detections in PASCAL format and score
fid = fopen(sprintf(VOCopts.detrespath, 'comp3', cls), 'w');
for i = 1:length(ids);
  bbox = boxes{i};
  for j = 1:size(bbox,1)
    fprintf(fid, '%s %f %d %d %d %d\n', ids{i}, bbox(j,end), bbox(j,1:4));
  end
end
fclose(fid);

VOCopts.testset = testset;
[recall, prec, ap] = VOCevaldet(VOCopts, 'comp3', cls, true);

% force plot limits
ylim([0 1]);
xlim([0 1]);

% save results
save([cachedir cls '_pr_' testset], 'recall', 'prec', 'ap');
print(gcf, '-djpeg', '-r0', [cachedir cls '_pr_' testset '.jpg']);
