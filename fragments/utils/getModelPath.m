function modelPath = getModelPath(globalOpts,num_iter)
modelPath = strrep(globalOpts.modelPath,'.mat',['_' sprintf('%03.0f',num_iter) '.mat']);