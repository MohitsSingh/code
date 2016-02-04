function B = allSubsets(n)
B = dec2bin(0:2^n-1);
BB = repmat(' ',size(B,1),2*size(B,2));
BB(:,1:2:end) = B;
B = str2num(BB);
end