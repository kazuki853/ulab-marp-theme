---
marp: true
theme: theme_U
paginate: true
footer: "2023/05/22<span class='fcenter'>内田研究室 論文紹介</span>"
size: 4:3
math: katex
---

<!-- _class: cover-->
<!-- _paginate: false-->
<!-- class: fs28 -->

# Amortized Inference for<br>Causal Structure Learning

<span style="text-align: center; margin-top: 10px;">

##  因果構造学習のための償却推論

</span>

内田研究室 M1
中本 一輝

---

<!-- _class: fs32 -->

# 読んだ理由

- 因果探索が気になっていたから

- NeurIPSの新しい論文

- （ちょっと関連がある）VAEの論文を読んだことがあった

---

<!-- _class: fs32 -->

# 研究概要

- 事後分布を近似する方法である償却推論を因果探索と組み合わせた

- 償却推論で用いるNNを、因果探索の問題設定に合わせて構成している

- シミュレーションで作成したデータセットで有効性を検証


---

<!-- _class: section_title -->
# 背景：因果探索

<style>
section.section_title h1 {
    /
    /* background-color: red; */
}
</style>

---

# 例：チョコレートとノーベル賞

<div class="v_center h_center">
<img src="images/choco_novel.png" height="500px">
</div>

---

<div class="v_center h_center">
<img src="images/choco_novel_causal.png" height="500px">
</div>

---

# 因果グラフ

<div class="split">
<div class="h_center split_l v_center" style="flex-basis: 30%;">
<img src="images/causal_graph.png" height="140px">
</div>

<div class="split_r" style="flex-basis: 80%;">

- **観測変数**（**observed variable**）
    データが収集される変数。四角で囲まれる。
- **未観測変数**（**unobserved variable**）
    データが収集されない変数。点線の楕円で囲まれる。

</div>
</div>

矢印の始点が原因の変数で、終点が結果の変数となっている。
このような定性的な因果関係を表す図を、**因果グラフ**（**causal graph**）という。
※因果効果の大きさなどの定量的な情報は含まれない。


## では、そもそもどのようなときに因果関係があるといえるのか？
## → 反事実モデルによって定義

---

# 反事実モデル（個体レベルの因果）

個体Aはある病気にかかっている。我々は、ある薬が個体Aの病気を治すかどうか知りたい。
→ 個体Aの2つの行動の結果を比較する。
<div class="h_center">
<img src="images/unit-level.png" height="350px">
</div>

- 片方を観測してしまうと、もう片方は観測できない（**因果推論の根本問題**）

---

# 反事実モデル（集団レベルの因果）

ある集団のすべての個体がある病気にかかっていたとする。
<div class="h_center">
<img src="images/population-level.png" height="400px">
</div>

集団について考える因果関係を、**集団レベルの因果**（**population-level causation**）という。

---

<!-- _class: fs24 -->

# 構造的因果モデル（SCM）

**構造的因果モデル**（**structual causal model**）
　　　　　＝　反事実モデル　＋　構造方程式モデル

（構造方程式モデル：データ生成過程を決定的に表したもの）

<div class="card title">

**定義　介入**

ある変数$x$に**介入する**とは、「他の変数がどんな値をとろうとも、変数$x$の値を定数$c$にとる」ことを意味する。

他の変数とは、観測される変数も、されない変数も含めたすべての変数。

このような介入を、$\mathrm{do}$という記号を用いて$\mathrm{do}(x=c)$と表す。
</div>

- 病気の薬の例で言えば、$x$に介入するとは、「年齢や性別、重症度に関わらず必ず薬を飲んでもらう」ということ。

---

# 構造的因果モデルでの因果の表現

<div class="h_center">
<img src="images/intervention_population-level.png" height="500px">
</div>

---

<!-- _class: fs24 -->

# とりあえず、この論文では…

$d$個の変数$\mathbf{x} = (x_1, \ldots, x_d)$の因果構造$G$とは、それぞれの辺が$\mathbf{x}$の変数間の因果効果を表した有向グラフ。

変数$x_i$が変数$x_j$に対して因果効果を持つとは、変数$x_i$への介入が変数$x_j$に、他の変数$\mathbf{x}_{\backslash ij} := \mathbf{x} \backslash \{x_i, x_j\}$とは独立に影響することである。つまり、
$$
p(x_j | \mathrm{do}(x_i = a, \mathbf{x}_{\backslash ij} = \mathbf{c})) \ne p(x_j | \mathrm{do}(x_i = a', \mathbf{x}_{\backslash ij} = \mathbf{c}))
$$
を満たす$a \ne a'$が存在することである。

- 因果構造$G$は非巡回だと嬉しい

---

<!-- _class: section_title -->

# 背景：償却（変分）推論

---

<!-- class: fs24 -->

# 償却推論とは

## 確率モデルにおける推論 = 事後分布の計算

でもモデルが複雑だと（解析的には）計算できない → **事後分布を近似**

1. ギブスサンプリング

1. 変分推論
    やっぱり計算が大変

## 償却推論では…
- 近似事後分布にパラメトリックな分布を仮定
- 近似分布のパラメータを、データから直接予測するモデル（関数）を作る
- 結局、この関数を最適化するという問題になる

---

# 償却推論の例：VAE

- 訓練データと似たような画像を作る生成モデルとして有名

<div class="h_center v_center">
<img src="images/vae.png" height="300px">
</div>

---

# VAEの仕組み（ざっくり）

<div class="h_center v_center">
<img src="images/vae2.jpeg" height="500px">
</div>

---

<!-- _class: section_title -->

# 提案手法
## AVICI: Amortized Variational Inference for Causal Discovery

---

# 問題設定

- データ生成分布：$p(D)$
- 観測データ：$D = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\} \sim p(D)$
- データ生成過程：$p(D|G)$

目標：観測データ$D$から因果構造の事後分布$p(G|D)$を$q(G; \theta)$で近似する

そこで、近似分布のパラメータ$\theta$を推論モデル$f_\phi$で予測する（償却推論）。
事後分布と近似分布のforward KLを最小化するように、$f_\phi$を学習する。
$$
\min_\phi \mathbb{E}_{p(D)} D_{\mathrm{KL}}(p(G|D) \| q(G; f_\phi(D)))
$$

---

# 目的関数

$$
\begin{aligned}
\mathbb{E}_{p(D)} D_{\mathrm{KL}}&(p(G|D) \| q(G; f_\phi(D))) \\
&= \mathbb{E}_{p(D)} \mathbb{E}_{p(G|D)}[\log p(G|D) - \log q(G; f_\phi(D))] \\
&= - \mathbb{E}_{p(G)} \mathbb{E}_{p(D|G)}[\log q(G; f_\phi(D))] + \mathrm{const.}
\end{aligned}
$$
定数の部分は$\phi$に依存しないので、$\mathcal{L}(\phi) := \mathbb{E}_{p(G)} \mathbb{E}_{p(D|G)}[\log q(G; f_\phi(D))]$を最大化すれば良い。

- 真のデータ生成分布$p(G,D)$からサンプルして、$q(G; \theta)$を予測する
- reverse KLの分散を小さく見積もってしまうという問題が起きないらしい
    - 変分推論ではよくreverse KLで最適化が行われる

<div class="memo">

reverse KL$D_{\mathrm{KL}}(q\|p)$には再構成誤差の項$\mathbb{E}_{q(G; \theta)}[\log p(D|G)]$が含まれる。しかし、周辺尤度$p(D|G)$の計算をするために、モデルの制約が増やす必要がある。その必要のない今回のモデルをLikelihood-Free Inferenceと呼んでいる。
</div>

---

# 近似分布

近似分布$q(G; \theta)$は次のようにベルヌーイ分布を用いる。
$$
q(G; \theta) = \prod_{i,j} q(g_{i,j}; \theta_{i,j}) \quad \mathrm{with} \quad g_{i,j} \sim \mathrm{Bern}(\theta_{i,j})
$$
推論モデル$f_\phi$は、$n$個のサンプル$\{\mathbf{o}^1, \ldots, \mathbf{o}^n\}$に対応するデータセット$D$に、$d \times d$行列を対応させる写像となる。

- それぞれのサンプル$\mathbf{o}^i = (o_1^i, \ldots, o_d^i)$には、観測された値$\mathbf{x}^i = (x_1^i, \ldots, x_d^i)$に加えて、その変数が介入を受けたかの情報も含まれる。

- 具体的には、$o_j^i = (x_j^i, u_j^i)$として、$u_j^i \in \{0, 1\}$はサンプル$i$において変数$j$が介入を受けたかを表す。

---

# 推論モデル$f_\phi$

推論モデル$f_\phi$には８層のニューラルネットワークを用いている。
具体的なNNの構成についてはよく理解できなかったので省略する。

- Transformerなどで用いられているmulti-head self-attentionが利用されているらしい

$f_\phi$はサンプルの順番や特徴量の並び順によって予測が変わってほしくない。
→ Max Poolingによってこの条件を達成している。

ネットワークの出力を$\mathbf{u}^i, \mathbf{v}^i \in \mathbb{R}^k$とすると、因果グラフの辺が存在する確率$\theta_{i,j}$は次のように決まる。
$$
\theta_{i,j} = \sigma(\tau \mathbf{u}^i \cdot \mathbf{v}^i + b)
$$
ここで、$\sigma$はロジスティック関数。

---

# 非巡回の制約
**特定のドメインでは因果グラフが非巡回という仮定をつけるとよく予測できる。**

このような制約は、$\phi$の制約として次のように書くことができる。
$$
\mathcal{F}(\phi) := \mathbb{E}_{p(D)}[h(f_\phi(D))] = 0
$$
※$h$の中身については省略

$\mathcal{F}(\phi) = 0$のもとで$\mathcal{L}(\phi)$を最大化するので、次のような問題を解けば良い。
$$
\min_\lambda \max_\phi \mathcal{L}(\phi) - \lambda \mathcal{F}(\phi)
$$

---

# 最適化アルゴリズム

<div class="h_center v_center">
<img src="images/alg1.png" height="300px">
</div>

- 非巡回の制約をつけない場合は$\lambda = 0$で開始する

---

<!-- _class: section_title -->
# 実験

---

# データセット

**真の因果構造がわかっている現実のデータセットが存在しない！**

この論文では３つの人工データを使って検証している。それぞれ、o.o.dやノイズが乗っているデータの場合も調べている

- SCM with Linear functions (LINEAR)

- SCM with nonlinear functions of random Fourier feature (RFF)

↑このふたつは構造因果モデルをもとにしたシミュレーション。
　因果構造にはスケールフリーネットワークを利用している。

- semisyntheticsingle-cell expression data of gene regulatory networks (GRNs)
    - 細胞内の確率的な遺伝子発現のシミュレータらしい（よくわからん）

---

# 評価指標

- 構造ハミング距離（SHD）：グラフ間の編集距離

- 構造介入距離（SID）：グラフの近さの定量化

- single-edge precision, recall, and F1 score

- AUPRC, AUROC

---

# 結果：OODデータに対する性能
<div class="h_center v_center">
<img src="images/fig1.png" height="400px">
</div>

- LEINEARとRFFでは提案手法は良い性能を発揮できる
- GRNではむしろOODデータのほうがうまく予測できる

---
<div class="h_center v_center">
<img src="images/fig1.png" height="450px">
</div>

下段：
- データ数を増やすと性能は上がるが、限界がある
- 変数が増えてタスクの難易度が上がると、なめらかに性能が下がる

---

# 結果：他の手法との比較
<div class="h_center v_center">
<img src="images/table1.png" height="350px">
</div>

- 多くの場合に既存のモデルよりも良い性能を達成している

---

<!-- _class: fs32 -->

# 所感

- VAEの仕組みは知っていたけど、償却推論は知らなかった
    - VAEなどでうまくいっている方法だけあって強い

- 検証できるデータセットが現実にないのは難しい
    
    - 生物分野のシミュレータが使われていて面白い

- グラフを直接NNに予想させるというのが新鮮だった

    - GNNなど、グラフを学習するだけだと思っていた
