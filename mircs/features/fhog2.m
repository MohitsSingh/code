function x = fhog2(I,T)
    if (nargin < 2)
        T = 8;
    end
    x = fhog(I,T);
    x = x(:,:,1:31);
end