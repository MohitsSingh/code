function [lineness,symmetry,M,orientation] = getLineProps(II)
[M,orientation] = gradientMag( im2single(II),1,0,0,0);
II = rgb2gray(II);
symmetry = phasesym(II);
[pc or phaseang T] = phasecongmono(II);

negphase = phaseang<0;
phaseang = negphase.*(-phaseang) + ~negphase.*phaseang;
% Then map angles > pi/2 to 0-pi/2
x = phaseang>(pi/2);
phaseang = x.*(pi-phaseang) + ~x.*phaseang;
lineness = (pi/2-phaseang)/(pi/2);
end