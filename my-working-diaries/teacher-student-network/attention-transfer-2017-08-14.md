## Paying More Attention to Attention: Improving the Performance of Convolutional Neural Network via Attention Transfer


### Derivation of Activation-based Attention Transfer

Let us consider a CNN layer and its corresponding activation tensor $A\in R^{C\times H\times W}$, which consists of $C$ feature planes with spatial dimensions $H\times W$.

Let $S,T$, and $W_S, W_T$ denote student, teacher and their weights correspondingly.

Let $\mathcal{L}(W, x)$ denote a standard cross entropy loss.

Let also $\mathcal{I}$ denote the indices of all teacher-student activation layer pairs for which we want to transfer attention maps.

<font color=red>**Forward Pass**</font>

* **Step 1**:
	Reshaping A such that $A\in R^{C\times (H*W)}$

* **Step 2**:
	$$Q = \mathcal{F}_{sum}^2(A) = \sum_{i=1}^{C}|A_{i,:}|^2, ~~~Q\in R^{H*W}.$$

* **Step 3**:
	Calculating $Q_S^j$ and $Q_T^j$ for all $j\in \mathcal{I}$, then for each $Q^k \in \{Q_S, Q_T\}$, L2-normalizing it
	$$Y^k = \frac{Q^k}{||Q^k||_2}.$$

* **Step 4**:
	Caculating
	$$Y^j = Y_S^j - Y_T^j$$
	for all $j\in \mathcal{I}$.

* **Step 5**:
	The final loss for attention maps is
	$$\mathcal{L_{a}} = \frac{\beta}{2}\sum_{j\in\mathcal{I}}||Y^j||_2$$

<font color=red>**Backward Pass**</font>

Note that $\mathcal{L}_a$ is a scalar.

* **Step 1**:
	$$\mathcal{L_{a}} = \frac{\beta}{2}\sum_{j\in\mathcal{I}}||Y^j||_2.$$
	$$\nabla_{Y^j}\mathcal{L_a} = \frac{\beta}{2}\frac{Y^j}{||Y^j||_2}.$$

* **Step 2**:
	$$Y^j = Y_S^j - Y_T^j.$$
	$$\nabla_{Y^j_S}Y^j = I.$$
	$$\nabla_{Y_S^j}\mathcal{L_a} =\nabla_{Y^j}\mathcal{L_a}\cdot\nabla_{Y^j_S}Y^j = \frac{\beta}{2}\frac{Y^j}{||Y^j||_2}.$$

* **Step 3**:
	$$Y_S^k = \frac{Q_S^k}{||Q_S^k||_2}, ~k\in\mathcal{I}.$$

	$$
	\frac{\partial (Y_S^k)_i}{\partial (Q_S^k)_j} =\left\{
	\begin{align}
	\frac{1}{(||Q_S^k||_2} .- \frac{(Q_S^k)_j^2}{(||Q_S^k||_2)^3}, ~~&if~i=j,\\
	- \frac{(Q_S^k)_i(Q_S^k)_j}{(||Q_S^k||_2)^3}, ~~&otherwise.
	\end{align}
	\right.
	$$

	$$\nabla_{Q_S^k} Y_S^k = \left[
	\begin{array}{rcl}
	\frac{1}{(||Q_S^k||_2} .- \frac{(Q_S^k)_1^2}{(||Q_S^k||_2)^3} & - \frac{(Q_S^k)_1(Q_S^k)_2}{(||Q_S^k||_2)^3}  & - \frac{(Q_S^k)_1(Q_S^k)_3}{(||Q_S^k||_3)^3} & ...\\
	- \frac{(Q_S^k)_2(Q_S^k)_1}{(||Q_S^k||_2)^3}  & \frac{1}{(||Q_S^k||_2} .- \frac{(Q_S^k)_2^2}{(||Q_S^k||_2)^3} &  - \frac{(Q_S^k)_2(Q_S^k)_3}{(||Q_S^k||_3)^3} & ...\\
	... & ... &... &...
	\end{array}\right]
	$$

* **Step 4**:
	$$Q = \mathcal{F}_{sum}^2(A) = \sum_{i=1}^{C}|A_{i,:}|^2, ~~~Q\in R^{H*W}.$$
 	$$\nabla_AQ = 2 * A \in R^{C\times(H*W)}$$

* **Step 5**:
	Reshaping $\nabla_AQ$ back to shape $R^{C\times H\times W}$.


<font color=red>**Caffe Implementation**</font>

* **Step 1 -- Reshaping**: `ReshapeLayer`
* **Step 2 -- Absolute**: `AbsValLayer`, computes $y = |x|$.
* **Step 3 -- Power**: `PowerLayer`, computes $y = (\alpha x + \beta)^{\gamma}$, as specified by the scale $\alpha$, shift $\beta$, and power $\gamma$.
* **Step 4 -- Convolution**: `ConvolutionLayer`, kernel size = 1.

Now, we have $Q = \mathcal{F} = vec(\sum_{i=1}^C(\alpha |A_{i,:}| + \beta)^{\gamma})$. Letting $\alpha = 1, \beta = 0$ gives $Q = vec(\sum_{i=1}^C|A_{i,:}|^{\gamma})$.

* **Step 5 -- Normalize**: `NormalizeLayer`, <font color=red> self-defined. Caffe-ssd has an implementation.</font> 

Now, we have $Q = \frac{Q}{||Q||_p}$.

* **step 6 -- AttentionLoss**: `AtLossLayer`, <font color=red> self-defined. </font> Maybe we can use `EuclideanLossLayer`, which computes $E = \frac{1}{2N}\sum_{n=1}^N||\hat{y}_n-y_n||_2^2$.


<font color="red"> **Model prototxt** </font>
```prototxt
layer{
	name: "reshape#i"
	bottom: "group#i/output"
	top: "group#i/output-reshaped"
	reshape_param{
		shape{
			dim: 0
			dim: 0
			dim: -1
			dim: 1
		}
	}
}

layer{
	name: "absolute_value"
	
}
```