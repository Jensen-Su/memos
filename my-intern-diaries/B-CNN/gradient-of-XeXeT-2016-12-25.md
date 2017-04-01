$$
  \begin{align}
    Cov(X) = E(XX^T) - EX(EX)^T &= \frac{1}{n-1}(XX^T - nEX(EX)^T)\\
    &= \frac{1}{n-1}(XX^T - \frac{1}{n}(Xe)(Xe)^T)
  \end{align}
$$

assume $Y = (Xe)(Xe)^T$ and $z = f(Y)$,

$$
\begin{align}
    \nabla_X z &= (\nabla_Y z) \nabla_X Y\\
               &= (\nabla_Y z) \nabla_X (Xe)(Xe)^T\\
               &= 2(\nabla_Y z) X(ee^T)
\end{align}
$$

assume $Y= Cov(X)$ and $z = f(Y)$, then

$$\nabla_Xz = \frac{2}{n-1}(\nabla_Yz)\left(X-\frac{1}{n}X(ee^T)\right)$$

where $e\in\{1\}^n$, $n$ is the number of columns of $X$.

For the case of B-CNN, do it on $X^T$.
