---
2016/11/25, Friday, Cloudy
---

    By Jincheng Su @ Hikvision, mail: jcsu14@fudan.edu.cn

Summary
---

### Add a self-defined layer `vl_weightedbilinearpool` to `matconvnet`

1. Define the layer `vl_weightedbilinearpool.m`

    The first step to add a self-defined layer to `matconvnet` is to define a function implementing the layer's functionalities of 'feed forward' and 'back propagation' with given inputs. '
    The following is what the `vl_weightedbilinearpool.m` may look like.

	```[matlab]
	% /path/to/bcnn/bcnn-package/vl_weightedbilinearpool.m

	function [y, varargout] = vl_weightedbilinearpool(x1, x2, W, varargin)
	% VL_WEIGHTEDBILINEARPOOL implementes the revised bilinear model with a weights matrix
	%
	% Copyright (C) 2016 Jincheng Su @ Hikvision.
	% All rights reserved.
	%
	% * **Feed forward**
	%   * Input: `x1` and `x2`, with shapes `[h1, w1, ch1, bs]` and `[h2, w2, ch2, bs]`.
	%            `W`, the weights
	%   * Output: `y`, with shape `[ch1*ch2, bs]`.
	%
	% * **Back propagation**
	%   * Input: `x1`, `x2` and `W` are the same as in forward pass, 
	%            dzdy = varargin{1}, is the derivative of loss `z` w.r.t `y`.
	%   * Output: y, the derivative of loss `z` w.r.t `x1`, i.e., dzdx1.
	%             varargout{1} = y2, the derivative of loss `z` w.r.t `x2`, i.e., dzdx2.
	%             varargout{2} = dw, the derivative of loss `z` w.r.t `W`, i.e., dzdW.
	%
	% -----------------------
	% The revised B-CNN model
	% -----------------------
	%
	%  $$ Y = X_a^TWX_b,$$
	%
	% where
	% * $X_a$ is with size $P_a\times CH_a,$
	% * $X_b$ is with size $P_b\times CH_b,$
	% * $W~$ is with size $P_a\times P_b,$
	% * $Y~~$ is with size $CH_a\times CH_b$.

	% The derivatives w.r.t $X_a$, $X_b$, and $W$ are
	% * $\nabla_{X_a}Z = WX_b\cdot(\nabla_YZ)^T$, $~~~~$(size: $P_a\times P_b \cdot P_b\times CH_b\cdot CH_b\times CH_a = P_a\times CH_a$)
	% * $\nabla_{X_b}Z = W^TX_a\cdot(\nabla_YZ)$,  $~~~~$(size: $P_b\times P_a \cdot P_a\times CH_a\cdot CH_a\times CH_b = P_b\times CH_b$)
	% * $\nabla_{W}Z = X_a\cdot(\nabla_YZ)\cdot X_b^T$.  $~~~$(size: $P_a\times CH_a \cdot CH_a\times CH_b\cdot CH_b\times P_b = P_a\times P_b$)
	%
	% -------------------------------------------------------------------------------------------------------

	% flag for doing backward pass
	isBackward = numel(varargin) > 0 && ~isstr(varargin{1});
	if isBackward
		dzdy = varargin{1};
	end

	% if GPU is used
	gpuMode = isa(x1, 'gpuArray');

	% [height, widht, channels, batchsize]
	[h1, w1, ch1, bs] = size(x1);
	[h2, w2, ch2, ~ ] = size(x2);

	if ~isBackward
		% forward pass
		if gpuMode
			y = gpuArray(zeros([1, 1, ch1 * ch2, bs], 'single'));
		else
			y = zeros([1, 1, ch1 * ch2, bs], 'single');
		end

		for b = 1: bs
			Xa = reshape(x1(:,:,:,b), [h1 * w1, ch1]);
			Xb = reshape(x2(:,:,:,b), [h2 * w2, ch2]);

			y(1, 1, :, b) = reshape(Xa'*W*Xb, [1, ch1 * ch2]); %'
		end
	else
		% backward pass
		if gpuMode
			y1 = gpuArray(zeros(h1, w1, ch1, bs, 'single')); 
			y2 = gpuArray(zeros(h2, w2, ch2, bs, 'single')); 
			dw = gpuArray(zeros(h1*w1, h2*w2,'single')); 
		else
			y1 = (zeros(h1, w1, ch1, bs, 'single')); 
			y2 = (zeros(h2, w2, ch2, bs, 'single')); 
			dw = (zeros(h1*w1, h2*w2,'single')); 
		end

		for b = 1: bs
			dZdY = reshape(dzdy(1, 1, :, b), [ch1, ch2]);
			Xa = reshape(x1(:,:,:,b), [h1 * w1, ch1]);
			Xb = reshape(x2(:,:,:,b), [h2 * w2, ch2]);
			dZdXa = reshape(W*Xb*dZdY', [h1, w1, ch1]);
			dZdXb = reshape(W'*Xa*dZdY, [h2, w2, ch2]);
			dZdW = Xa*dZdY*Xb'; %'

			y1(:, :, ;, b) = dZdXa;
			y2(:, :, :, b) = dZdXb;
			dw = dw + dZdW;
		end
		y = y1;
		varargout{1} = y2;
		varargout{2} = dw / bs;

	end
	```
2. Register the layer to `vl_simplenn.m`.
    Since I always hold the belief of never changing the source code easily, I would leave `vl_simplenn.m` untouched but rather revise its alternative `vl_bilinearnn.m`. 

        $ cp /path/to/bcnn/bcnn-package/vl_bilinearnn.m /path/to/bcnn/bcnn-package/vl_bilinearnn.m_bak
        $ gvim /path/to/bcnn/bcnn-package/vl_bilinearnn.m
    ---
    ```[matlab]
    % vl_bilinearnn.m

        ...
        % forward pass
            case 'pdist'
              res(i+1) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
            case 'bilinearpool'
              res(i+1).x = vl_nnbilinearpool(res(i).x);
            case 'bilinearclpool'
              x1 = res(l.layer1+1).x;
              x2 = res(l.layer2+1).x;
              res(i+1).x = vl_nnbilinearclpool(x1, x2);

            % this is my layer. As a first try, I simply assume the input `x1` and `x2` are the same
            case 'weightedbilinearpool'
              res(i+1).x = vl_weightedbilinearpool(res(i).x, res(i).x, l.weights{1});

            case 'sqrt'
              res(i+1).x = vl_nnsqrt(res(i).x, 1e-8);
            case 'l2norm'
              res(i+1).x = vl_nnl2norm(res(i).x, 1e-10);
            case 'custom'

        ...

        % backward pass
            case 'bilinearclpool'
                x1 = res(l.layer1+1).x;
                x2 = res(l.layer2+1).x;
                [y1, y2] = vl_nnbilinearclpool(x1, x2, res(i+1).dzdx);
                res(l.layer1+1).dzdx = updateGradient(res(l.layer1+1).dzdx, y1);
                res(l.layer2+1).dzdx = updateGradient(res(l.layer2+1).dzdx, y2);

            % this is my layer. As a first try, I simply assume the input `x1` and `x2` are the same
            case 'weightedbilinearpool'
                [y1, y2, dzdw{1}] = vl_weightedbilinearpool(res(i).x, res(i).x, l.weights{1}, res(i+1).dzdx);
                res(i).dzdx = updateGradient(res(i).dzdx, y1 + y2);
                clear y1 y2

            case 'sqrt'
                backprop = vl_nnsqrt(res(i).x, 1e-8, res(i+1).dzdx);
                res(i).dzdx = updateGradient(res(i).dzdx, backprop);
                clear backprop
        ...

        % Add our type. 
        switch l.type
            case {'conv', 'convt', 'bnorm', 'weightedbilinearpool'}
                if ~opts.accumulate
                    res(i).dzdw = dzdw ;
                else
                    for j=1:numel(dzdw)
                        res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                end
            end
            dzdw = [] ;
        end

    ```
3. Our work of adding a layer to `matconvnet` is almost done. However, one another small revision is indispensable for enabling our layer run on a GPU unbuggly when using `vl_bilinearnn.m`. That is, add our layer to the `vl_simplenn_move.m`.

	```[matlab]
	% /path/to/bcnn/matconvnet/matlab/simplenn/vl_simplenn_move.m
	for l=1:numel(net.layers)
		switch net.layers{l}.type
			case {'conv', 'convt', 'bnorm', 'weightedbilinearpool'}
				for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
	```

### Build our `WB-CNN` model.
    Now that we have added a self-defined layer with type `weightedbilinearpool` to `matconvnet`, we need to build a network to test if it works correct or if it ever works. I'll do it by revised the `B-CNN` model.
    As a first try, let's implementes a simple symmetric `WB-CNN`, which can be built easily by replace the `bilinearpool` layer of a symmetric `B-CNN` with our `weightedbilinearpool` layer.  
    1. Modify `initializeNetworkSharedWeights.m`
        $ cp /path/to/bcnn/initializeNetworkSharedWeights.m /path/to/bcnn/initializeNetworkWeightedBcnn.m
        $ gvim /path/to/bcnn/initializeNetworkWeightedBcnn.m

    ```[matlab]
    % initializeNetworkWeightedBcnn.m
        
        ...

        % build a linear classifier netc
        netc.layers = {};

        h1 = 27;
        w1 = 27;
        bilinearW = 0.001/scal * randn(h1*w1, h1*w1, 'single');
        % stack weighted bilinearpool layer
        netc.layers{end+1} = struct('type', 'weightedbilinearpool', 'name', 'wblp', ...
            'weights',{{bilinearW}} , ...
            'learningRate', [1000], ...
            'weightDecay', [0.9]);

        % stack a dropout layer, `rate` is defined as a probability of a varaiable not to be zeroed.
        netc.layers{end+1} = struct('type', 'dropout', 'name', 'dropout_wbcnn',...
            'rate', 0.3);

        % stack a relu layer
         netc.layers{end+1} = struct('type', 'relu', 'name', 'relu6');

        % stack normalization
        netc.layers{end+1} = struct('type', 'sqrt', 'name', 'sqrt_norm');
        netc.layers{end+1} = struct('type', 'l2norm', 'name', 'l2_norm');

        % stack classifier layer
        initialW = 0.001/scal * randn(1,1,ch1*ch2,numClass,'single');
        initialBias = init_bias.*ones(1, numClass, 'single');
        netc.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
            'weights', {{initialW, initialBias}}, ...
            'stride', 1, ...
            'pad', 0, ...
            'learningRate', [1000 1000], ...
            'weightDecay', [0 0]) ;
        netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
        netc = vl_simplenn_tidy(netc) ;

        ...

                    codeb = squeeze(gather(res(end).x));
                    for i=1:numel(batch)
                         %
                         code = codeb(:, :, :, i);
        %                code = reshape(codeb(:, :, :, i), size(codeb, 1)*size(codeb, 2), size(codeb, 3));               
                        savefast(fullfile(opts.nonftbcnnDir, ['bcnn_nonft_', num2str(batch(i), '%05d')]), 'code');
                    end
                end
            end

        ...

        function [im,labels] = getBatch_bcnn_fromdisk(imdb, batch)
        % -------------------------------------------------------------------------

        imtmp = cell(1, numel(batch));
        for i=1:numel(batch)
            load(fullfile(imdb.imageDir, imdb.images.name{batch(i)}));
            imtmp{i} = code;
        end
        h = size(imtmp{1}, 1);
        w = size(imtmp{1}, 2);
        ch = size(imtmp{1}, 3);
        im = zeros(h, w, ch, numel(batch));
        for i = 1:numel(batch)
            im(:, :, :, i) = imtmp{i};
        end
        clear imtmp
        %im = cat(2, im{:});
        %im = reshape(im, size(im, 1), size(im, 2), size(im,3), size(im, 4));
        labels = imdb.images.label(batch) ;
    ```

    2. Open `/path/to/bcnn/matconvnet/examples/cnn_train.m` and replace the `vl_simplenn` with `vl_bilinearnn`.
    ```[matlab]
    %     res = vl_simplenn(net, im, dzdy, res, ...
    %                       'accumulate', s ~= 1, ...
    %                       'mode', evalMode, ...
    %                       'conserveMemory', params.conserveMemory, ...
    %                       'backPropDepth', params.backPropDepth, ...
    %                       'sync', params.sync, ...
    %                       'cudnn', params.cudnn, ...
    %                       'parameterServer', parserv, ...
    %                       'holdOn', s < params.numSubBatches) ;
                      
         res = vl_bilinearnn(net, im, dzdy, res, ...
                          'accumulate', s ~= 1, ...
                          'mode', evalMode, ...
                          'conserveMemory', params.conserveMemory, ...
                          'backPropDepth', params.backPropDepth, ...
                          'sync', params.sync, ...
                          'cudnn', params.cudnn) ;
    ```


