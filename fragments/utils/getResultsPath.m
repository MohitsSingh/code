function resultsPath = getResultsPath(globalOpts,numIter)
resultsPath = [globalOpts.resultPath '_' sprintf('%03.0f',numIter)];
end