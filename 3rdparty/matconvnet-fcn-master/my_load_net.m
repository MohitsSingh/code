function [net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(modelPath,modelFamily)

switch modelFamily
    case 'matconvnet'
        net = load(modelPath) ;
        net = dagnn.DagNN.loadobj(net.net) ;
        net.mode = 'test' ;
        for name = {'objective', 'accuracy'}
            net.removeLayer(name) ;
        end
        net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,[]) ;
        predVar = net.getVarIndex('prediction') ;
        inputVar = 'input' ;
        imageNeedsToBeMultiple = true ;
        
        
    case 'ModelZoo'
        net = dagnn.DagNN.loadobj(load(modelPath)) ;
        net.mode = 'test' ;
        predVar = net.getVarIndex('upscore') ;
        inputVar = 'data' ;
        imageNeedsToBeMultiple = false ;
        
    case 'TVG'
        net = dagnn.DagNN.loadobj(load(modelPath)) ;
        net.mode = 'test' ;
        predVar = net.getVarIndex('coarse') ;
        inputVar = 'data' ;
        imageNeedsToBeMultiple = false ;
end
end

