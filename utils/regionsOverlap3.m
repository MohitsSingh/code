function [ovps ints uns] = regionsOverlap3(regions1,regions2)
    n1 = length(regions1);
    n2 = length(regions2);
    if n1 == 0 || n2 == 0 
        ovps = zeros(n1,n2);
        ints = zeros(n1,n2);
        uns = zeros(n1,n2);
        return;
    end
      
    [ints,a1,a2] = regionsInt(regions1,regions2);
    
   
    [a1,a2] = meshgrid(a2,a1);
    uns = a1+a2-ints;
    ovps = ints./uns;    
end