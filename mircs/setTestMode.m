
function params = setTestMode(params,testMode, testModes)
params.testMode = testMode;
if nargin < 3
    testModes = repmat(testMode,size(params.phases));
end
for i = 1:length(params.phases)
    params.phases(i).alg_phase.setTestMode(testModes(i));
end
