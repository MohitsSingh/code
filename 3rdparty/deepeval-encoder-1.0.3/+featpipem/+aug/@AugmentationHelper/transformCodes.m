function code = transformCodes(obj, codes)
    
    run_config = obj.get_run_config();

    if ~strcmp(run_config.augmentation, 'none')
        if strcmp(run_config.augmentation_collate, 'sum') || ...
                strcmp(run_config.augmentation_collate, 'sumraw')
            codes = norm_mat(codes, obj.prepool_norm_type);
            codes = mean(codes, 2);
        elseif strcmp(run_config.augmentation_collate, 'max') || ...
                strcmp(run_config.augmentation_collate, 'maxraw')
            codes = norm_mat(codes, obj.prepool_norm_type);
            codes = max(codes, [], 2);
        elseif strcmp(run_config.augmentation_collate, 'stack')
            codes = norm_mat(codes, obj.subcode_norm_type);
            
            assert(size(codes,2) == obj.get_output_dim_mul());
            codes = reshape(codes, [size(codes,1)*size(codes,2), 1]);
        elseif ~strcmp(run_config.augmentation_collate, 'none')
            error('Unknown collation type');
        end
    end

    % Normalize -----------------------------------------------------------
    
    codes = kermap_mat(codes, obj.kermap);
    if ~strcmp(run_config.augmentation_collate, 'maxraw') && ...
            ~strcmp(run_config.augmentation_collate, 'sumraw')
        code = norm_mat(codes, obj.norm_type);
    else
        code = codes;
    end
    
    function normed_mat = norm_mat(mat, norm_type)

        if strcmp(norm_type, 'l1')
            mat_norm = sum(mat, 1);
            mat_norm(mat_norm < eps) = eps;
            normed_mat = bsxfun(@rdivide, mat, mat_norm);
        elseif strcmp(norm_type, 'l2')
            mat_norm = sqrt(sum(mat.^2, 1));
            mat_norm(mat_norm < eps) = eps;
            normed_mat = bsxfun(@rdivide, mat, mat_norm);
        elseif strcmp(norm_type, 'none') || isempty(norm_type)
            normed_mat = mat;
        else
            error('Unrecognized norm type');
        end
        
    end
    
    function kermapped_mat = kermap_mat(mat, kermap)
        
        if strcmp(kermap, 'hellinger')
            kermapped_mat = sign(mat) .* sqrt(abs(mat));
        elseif strcmp(kermap, 'none')
            kermapped_mat = mat;
        else
            error('Unrecognised kernel map %s!', kermap);
        end
    end

end