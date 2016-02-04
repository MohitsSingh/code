function B = bwtraceboundary2(R)
[ii,jj] = find(R,1,'first');
B = bwtraceboundary(R,[ii jj],'W');