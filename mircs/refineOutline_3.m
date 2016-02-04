function mask = refineOutline_3(I,mask,s)
 
 d_out = bwdist(mask);
 d_in = bwdist(~mask);
%  obj_mask = zeros(size(mask));
 obj_mask = double(mask)*.5;
 
%  inside = d_in>=s;
%  outside = d_out>=s;
%  
%  Z = zeros(size(mask));
% Z(~inside & ~outside) = sigmoid(-d_out(~inside & ~outside));
obj_mask(d_in >=s)=1;
obj_mask(d_in < s/2 & d_out < s) = .5;

obj_box = region2Box(mask);
obj_box = round(inflatebbox(obj_box,2));
I_sub = cropper(I,obj_box);
obj_mask = cropper(obj_mask,obj_box);
 
%  obj_mask(d_in < s & d_out < s)
[seg_mask,energies] = st_segment(im2uint8(I_sub),obj_mask,.5,5);
% clf;displayRegions(I_sub,seg_mask);
% b = bwtraceboundary2(cropper(mask,obj_box));
% plotPolygons(fliplr(b),'g--');
mask = shiftRegions(seg_mask,obj_box,I);

% clf;displayRegions(I,d_in>=s)
% clf;displayRegions(I,d_out<=s)

% gc_segResult = getSegments_graphCut_2(I_sub,obj_mask,[],0,obj_mask_hard);