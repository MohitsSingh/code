function ap = run

% ap = run
% run the detector on the test dataset and evaluate the results

boxes1 = test; 
disp('Testing done, now the boxes will be evaluated');
pause(1);
ap = evaluate_results(boxes1);

