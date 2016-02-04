function I = denormalise(I,I_min,I_max)
I = I*I_max;
I = I+I_min;
end