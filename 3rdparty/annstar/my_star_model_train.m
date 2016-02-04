function [params, sms, train_posXY, encoders] = my_star_model_train(input_train_imgs, train_annotations, resize_input)

    %% general
    addpath 'I:/code/vlfeat-0.9.18/apps/recognition';
    addpath 'I:/code/vlfeat-0.9.18/toolbox';
    vl_setup;

    if (resize_input)
        [train_imgs, train_data] = get_dataset(input_train_imgs, train_annotations);
    else
        train_imgs = input_train_imgs;
        train_data = train_annotations;
    end
    
    [params] = get_params();
    [sms] = model_train(params, train_imgs, train_data); 
    
    clear train_posXY;
    for i_type=1:params.n_descr_type
        train_posXY{i_type} = [cell2mat(sms{i_type}.posXY), cell2mat(sms{i_type}.src_img)];
    end

    args_kdtreebuild = {'NumTrees', 1};
    clear encoders;
    for i_type=1:params.n_descr_type
        clear encoder;
        encoder.words = cell2mat(sms{i_type}.descr)';
        encoder.kdtree = vl_kdtreebuild(encoder.words, args_kdtreebuild{:});
        encoders{i_type} = encoder;
    end
    

return;

function [imgs, dataset] = get_dataset(imgs_30x30, annotations)
    clear imgs
    for iImg=1:length(imgs_30x30)
        imgs{iImg} = imresize(imgs_30x30{iImg},2);
    end
    s = size(imgs{1},1);

    clear cntXY;
    for iImg=1:length(annotations)
        c = (fliplr(annotations{iImg}{1}{1}));
        c = round(s*c/108);
        cntXY{iImg}=c;
    end
    dataset.class.cntXY = cntXY;
return;


function [params] = get_params()
    step=2;
    
    i_type = 1;
    geom{i_type} = 4;%3;
    cellsize{i_type} = 4;
    edges_only{i_type} = false;
    descr_args = {'step', step, 'cellsize', cellsize{i_type}, 'scales', 1, ...
                    'geometry', [geom{i_type} geom{i_type}], 'edges_only', edges_only{i_type}};
    extractorFn{i_type}= @(x) getDenseHOG(x, descr_args{:});

%     i_type = i_type+1;
%     geom{i_type} = 5;
%     cellsize{i_type} = 4;
%     edges_only{i_type} = false;
%     descr_args = {'step', step, 'cellsize', cellsize{i_type}, 'scales', 1, ...
%                     'geometry', [geom{i_type} geom{i_type}], 'edges_only', edges_only{i_type}};
%     extractorFn{i_type}= @(x) getDenseHOG(x, descr_args{:});

%    i_type = i_type+1;
%    geom{i_type} = 4;
%    cellsize{i_type} = 4;
%    edges_only{i_type} = true;
%    descr_args = {'step', step, 'cellsize', cellsize{i_type}, 'scales', 1, ...
%                    'geometry', [geom{i_type} geom{i_type}], 'edges_only', edges_only{i_type}};
%    extractorFn{i_type}= @(x) getDenseHOG(x, descr_args{:});
    
%    i_type = i_type+1;
%    geom{i_type} = 3;
%    cellsize{i_type} = 2;
%    edges_only{i_type} = false;
%    descr_args = {'step', step, 'cellsize', cellsize{i_type}, 'scales', 1, ...
%                    'geometry', [geom{i_type} geom{i_type}], 'edges_only', edges_only{i_type}};
%    extractorFn{i_type}= @(x) getDenseLBP(x, descr_args{:});

    params.extractorFn = extractorFn;
    params.n_descr_type = length(params.extractorFn);
    params.geom = geom;
    params.cellsize = cellsize;
    params.edges_only = edges_only;
    params.tolerance = 4;

return;

function [sm] = model_train(params, imgs, train_data) 
    %% train the model
    n_descr_type = length(params.extractorFn);
    clear sm;
    for iImg=1:length(imgs)
        img=imgs{iImg};

        cntXY=train_data.class.cntXY{iImg};

        for i_type=1:n_descr_type

            features = params.extractorFn{i_type}(img);

            descr = features.descr';
            posXY = features.frame(1:2,:)';
            offs2cntXY=bsxfun(@minus,cntXY,posXY);

            sm{i_type}.descr{iImg,1}=descr;
            sm{i_type}.offs2cntXY{iImg,1}=offs2cntXY;
            sm{i_type}.src_img{iImg,1}=iImg*ones(size(posXY,1),1);
            sm{i_type}.posXY{iImg,1}=posXY;
        end
    end

return;