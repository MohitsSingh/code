function [test_estimatePosXY, test_support_map, res_data] = my_star_model_test(input_test_imgs, params, sms, train_posXY, encoders, input_train_imgs, resize_input)

    debug_show_me=1;
    if (debug_show_me)
        i_main_fig=1;
        i_second_fig=2;
    end


    if (resize_input)
        clear test_imgs
        for iTestImg=1:length(input_test_imgs)
            test_imgs{iTestImg} = imresize(input_test_imgs{iTestImg},2);
        end

        clear train_imgs
        for iTrainImg=1:length(input_train_imgs)
            train_imgs{iTrainImg} = imresize(input_train_imgs{iTrainImg},2);
        end
    else
        test_imgs = input_test_imgs;
        train_imgs = input_train_imgs;
    end
    n_train_imgs = length(train_imgs); 

    %%
    test_params.k=32;%25;
    test_params.sigma=0.2;
    test_params.geom_sigma_cnt=2*floor(params.tolerance/2)+1;%10;
    test_params.bpMult=1;

    clear res;
    for iTestImg=1:length(test_imgs)
        test_img=test_imgs{iTestImg};
        clear vms;
        clear cntVotes;
        clear idxVotes;
        clear posXYs;
        clear probs;

        for i_type=1:params.n_descr_type
            % vote for center
            clear sm;
            sm.offs2cntXY=cell2mat(sms{i_type}.offs2cntXY);
            test_params.extractorFn = params.extractorFn{i_type};
            [vms{1,1,i_type}, cntVotes{i_type}, idxVotes{i_type}, posXYs{i_type}, probs{i_type}] = star_model_test(encoders{i_type}, sm, test_img, test_params);
        end
 
        %find center location
        clear cPosXY;

        %sum all votes
        vm = sum(cell2mat(vms),3);
                
        %set center location to be the max of the heat map, ie the most probable location
        [cPosXY(2),cPosXY(1)]=find(vm==max(vm(:)));

        if (debug_show_me)
            figure(i_main_fig);
			rc = max(params.n_descr_type,3);
            subplot(2,rc,1); imshow(repmat(test_img,[1,1,3])); title(['image ' num2str(iTestImg)]);
            subplot(2,rc,2); imshow(repmat(single(edge(test_img,'canny',[],0.1)),[1,1,3])); title(['egdes ' num2str(iTestImg)]);
            subplot(2,rc,3); imshow(repmat(single(test_img)/max(single(test_img(:))),[1 1 3]));hold on;h=imagesc(vm);set(h,'AlphaData',0.3);title(['vote map' num2str(iTestImg)]);plot(cPosXY(1),cPosXY(2),'ko');
        end
       
        %start back projection
        clear bpVVs;
        clear vote_maps;
        for i_type=1:params.n_descr_type
            cntVote = cntVotes{i_type};
            cntVote_ = round(cntVote);
            vv = all(cntVote_>=1,3) & all(bsxfun(@le,cntVote_,reshape(fliplr(size(test_img)),[1,1,2])),3);

            %do back projection to keep only votes that supported the most probable center
            %NN who voted to a location near the most probable center are kept 
            bpVVs{i_type} = vv & (sqrt(sum(bsxfun(@minus,cntVote,reshape(cPosXY,[1,1,2])).^2,3))<=test_params.bpMult*test_params.geom_sigma_cnt);

            bpVV = bpVVs{i_type};
            prob = probs{i_type};
            idxVote = idxVotes{i_type};
            vote_maps{i_type}=accumarray(round(train_posXY{i_type}(idxVote(bpVV),:)),prob(bpVV),[size(test_img), n_train_imgs]);

            if (debug_show_me)
                posXY = round(posXYs{i_type});
                supportVote_x=repmat(posXY(:,1)',[test_params.k,1]);
                supportVote_y=repmat(posXY(:,2)',[test_params.k,1]);
                support_vm = (accumarray([supportVote_y(bpVV),supportVote_x(bpVV)],prob(bpVV),size(test_img)));
                %support_vm(support_vm~=ordfilt2(support_vm,((cellsize{i_type})^2),true(cellsize{i_type})))=0; %nms
                vals = sort(support_vm(:),'descend');
                support_vm(support_vm<vals(2*test_params.k))=0;
                support_vm=ordfilt2(support_vm,((params.cellsize{i_type}*params.geom{i_type})^2),true(params.cellsize{i_type}*params.geom{i_type}));
                if (params.edges_only{i_type})%(useEdgesOnly)
                   edge_img = edge(test_img,'canny',[],0.1);
                   support_vm =imdilate(edge_img,ones(round(params.cellsize{i_type}/2))).*support_vm;
                end
                figure(i_main_fig);
                subplot(2,rc,rc+i_type);
                imshow(repmat(single(test_img)/max(single(test_img(:))),[1 1 3]));hold on;h=imagesc(support_vm,[0,1]);set(h,'AlphaData',0.3); title(['center support ' num2str(i_type)]);
                iii=0;
            end
            iii=0;
        end
%         if (debug_show_me)
%             print(gcf,'-djpeg',sprintf('res/img%02d.jpg',iImg));
%         end

        clear img_support;
        for ihm=1:n_train_imgs
            support_hm = zeros(size(test_img));
            for i_type=1:params.n_descr_type
                hm = vote_maps{i_type}(:,:,ihm);
                hm = ((params.cellsize{i_type}*params.geom{i_type})^2)*imfilter(hm,fspecial('average',(params.cellsize{i_type}*params.geom{i_type})));
                if (params.edges_only{i_type})%(useEdgesOnly)
                   edge_img = edge(train_imgs{ihm},'canny',[],0.1);
                   hm =imdilate(edge_img,ones(round(params.cellsize{i_type}/2))).*hm;
                end
                support_hm = support_hm + hm;
            end
            img_support{1,1,ihm}=support_hm;
        end

        test_estimatePosXY(iTestImg,:) = cPosXY;
        test_support_map{iTestImg} = img_support;
    
        if (debug_show_me)
            figure(i_second_fig);
            rc=ceil(sqrt(4+n_train_imgs));
            subplot(rc,rc,[1,2,rc+1,rc+2]);
            imshow(repmat(single(test_img)/max(single(test_img(:))),[1 1 3])); title(['test image ' num2str(iTestImg)]);
            for ihm=1:n_train_imgs
                if (ihm<=rc-2)
                    subplot(rc,rc,2+ihm);
                else
                    subplot(rc,rc,4+ihm);
                end
                support_hm = img_support{1,1,ihm};
                imshow(repmat(single(train_imgs{ihm})/max(max(single(train_imgs{ihm}))),[1 1 3]));hold on;h=imagesc(support_hm,[0,1]);set(h,'AlphaData',0.3); %title(ihm);
            end
            %print(gcf,'-djpeg',sprintf('res/img%02d_vote.jpg',iImg));
        end

        %res_data{iTestImg}.cPosXY = cPosXY;
        %res_data{iTestImg}.support = support;
        %res_data{iTestImg}.vms = vms;
        res_data{iTestImg}.bpVVs = bpVVs;
        res_data{iTestImg}.probs = probs;
        res_data{iTestImg}.posXYs = posXYs;
        %res_data{iTestImg}.cntVotes = cntVotes;
        res_data{iTestImg}.idxVotes = idxVotes;
        
        iii=0;
    end


return;