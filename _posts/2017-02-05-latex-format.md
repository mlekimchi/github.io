---
layout: post
title: "Test LaTex File"
published: true
comments: true
---

{% latex %}

\begin{align}
f(x) & =\frac {1} {\sqrt {2 \pi} \sigma} \mathrm {e}^ \frac {-(x-\mu)^2} {2 \sigma ^2 } \\
F(x) & =\int _{-\infty} ^{x} \frac {1} {\sqrt {2 \pi} \sigma} \mathrm {e}^ \frac {-(t-\mu)^2} {2 \sigma ^2 } \mathrm {d} t
\end{align}
{% endlatex %}

{% latex %}
$$f(x) & =\frac {1} {\sqrt {2 \pi} \sigma} \mathrm {e}^ \frac {-(x-\mu)^2} {2 \sigma ^2 } $$

$$F(x) & =\int _{-\infty} ^{x} \frac {1} {\sqrt {2 \pi} \sigma} \mathrm {e}^ \frac {-(t-\mu)^2} {2 \sigma ^2 } \mathrm {d} t $$

{% endlatex %}