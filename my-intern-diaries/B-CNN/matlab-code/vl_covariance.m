function y = vl_covariance(X, varargin)
% -------------------------------------------------------------------------------------------------------
% VL_COVARIANCE implementes the revised bilinear model with a covariance
% instead
%
% Copyright (C) 2016.12.25 Jincheng Su @Fudan
% All rights reserved.
%
% Input:
%     X -- the a batch of feature map, with size [h, w, ch, bs]
%     varargin{1} -- dzdy when in backpropagation
% Output:
%     y -- the covariance in forward pass or dzdx in backward pass
%
% -----------------------
% The covariance model
% -----------------------
%
% $$
%   \begin{align}
%     Cov(X) = E(XX^T) - EX(EX)^T &= \frac{1}{n-1}(XX^T - nEX(EX)^T)\\
%     &= \frac{1}{n-1}(XX^T - \frac{1}{n}(Xe)(Xe)^T)
%   \end{align}
% $$
% 
% assume $Y = (Xe)(Xe)^T$ and $z = f(Y)$,
% 
% $$
% \begin{align}
%     \nabla_X z &= (\nabla_Y z) \nabla_X Y\\
%                &= (\nabla_Y z) \nabla_X (Xe)(Xe)^T\\
%                &= 2(\nabla_Y z) X(ee^T)
% \end{align}
% $$
% 
% assume $Y= Cov(X)$ and $z = f(Y)$, then
% 
% $$\nabla_Xz = \frac{2}{n-1}(\nabla_Yz)\left(X-\frac{1}{n}X(ee^T)\right)$$
% 
% where $e\in\{1\}^n$, $n$ is the number of columns of $X$.
%
% -------------------------------------------------------------------------------------------------------

% flag for doing backward pass
isBackward = numel(varargin) > 0 && ~isstr(varargin{1});
if isBackward
    dzdy = varargin{1};
end

% if GPU is used
gpuMode = isa(X, 'gpuArray');

% [height, widht, channels, batchsize]
[h, w, ch, bs] = size(X);

if ~isBackward
    % forward pass
    if gpuMode
        y = gpuArray(zeros([1, 1, ch * ch, bs], 'single'));
        e = gpuArray(ones([h*w, 1], 'single'));
    else
        y = zeros([1, 1, ch * ch, bs], 'single');
        e = ones([h*w, 1], 'single');
    end

    for b = 1: bs
        Xmat = reshape(X(:,:,:,b), [h * w, ch]);
        CovXt = (Xmat'*Xmat - (Xmat'*e)*(e'*Xmat)/ch)/(ch-1);  
        y(1, 1, :, b) = reshape(CovXt, [1, ch * ch]);% / (h1 * w1 ); %'
    end
else
    % backward pass
    if gpuMode
        y = gpuArray(zeros(h, w, ch, bs, 'single')); 
        e = gpuArray(ones([h*w, 1], 'single'));
    else
        y = (zeros(h, w, ch, bs, 'single'));
        e = ones([h*w, 1], 'single');
    end

    for b = 1: bs
        dy = reshape(dzdy(1, 1, :, b), [ch, ch]);
        Xmat = reshape(X(:,:,:,b), [h * w, ch]);
        dzdx = 2 * (Xmat -  (e*e')* Xmat/ ch) * dy'/(ch-1);   
        clear Xmat dy
        y(:, :, :, b) = reshape(dzdx, [h, w, ch]);
    end

end
