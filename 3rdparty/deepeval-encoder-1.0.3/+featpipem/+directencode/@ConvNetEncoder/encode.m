function code = encode(obj, im)

    assert(obj.net_handle_ > 0);
    assert(isa(im, 'single'));
    im = im*255.0;

    % get input images
    fixed_prep_image = (obj.caffe_version >= 1.1);

    input_data = ...
        {obj.augmentation_helper_.prepareImage(im, ...
                                               'mean_img', obj.average_image, ...
                                               'preproc_dup_grey', obj.preproc_dup_grey, ...
                                               'fixed_prep_image', fixed_prep_image)};

    % pass through net
    %fprintf('Forwarding with handle: %d\n', obj.net_handle_);
    code = caffe('forward', input_data, obj.output_blob_name, obj.net_handle_);
    code = code{1};

    net_code_dim = obj.get_net_output_dim_();
    code = reshape(code, [net_code_dim, size(code,1)/net_code_dim]);

    % collate and normalise result
    code = obj.augmentation_helper_.transformCodes(code);

end
