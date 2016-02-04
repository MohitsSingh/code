function MaskTarg = calc_mask_in_targ_image(SourceMask, TargImRows, TargImCols, shift_in_target_image)

[row, col] = find(SourceMask);

%clalc bounding box of mask in source image
start_bb = [min(col), min(row)];
end_bb   = [max(col), max(row)];
bb_size  = end_bb - start_bb;

if (bb_size(1) + shift_in_target_image(1) > TargImCols)
    shift_in_target_image(1) = TargImCols - bb_size(1);
end

if (bb_size(2) + shift_in_target_image(2) > TargImRows)
    shift_in_target_image(2) = TargImRows - bb_size(2);
end

%relocating source mask to mask in targ image 
MaskTarg = zeros(TargImRows, TargImCols);
MaskTarg(sub2ind([TargImRows, TargImCols], row - start_bb(2) + shift_in_target_image(2), col - start_bb(1) + shift_in_target_image(1))) = 1;
