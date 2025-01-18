[TOC]
#Datawhale组队学习

# Vision Transformer: Demystify and Build from Scratch

organized by Luyi Lin at Beijing Jiaotong University, 23271037@bjtu.edu.cn

## Preface

这是一趟

- 数学性
- 实践性
- 丰富：从不同角度进行理解，力求得到对Transformer深刻观察
- 前沿：Transformer^2^的引入 

## Seq2Seq

### Why we begin from Seq2Seq?
**Transformer**模型在Seq2Seq的基础上进行了重要的改进——Transformer抛弃了RNN和LSTM的结构，改为使用自注意力机制（Self-Attention），使得模型能够并行处理序列并捕捉长距离的依赖关系。从这里开始，可以在比较中慢慢掌握更复杂的模型设计。

### What is Seq2Seq

![alt text](C:\Users\14711\Desktop\Vision Transformer and Build from Scratch\image\image-4.png)
#### Origin
2014年，机器翻译任务 (Sutskever et al., 2014)[^1]中，为了解决输入与输出序列长短未知的问题，研究者引入了Seq2Seq模型

#### A Quick Flight
编码器将输入序列压缩为一个固定维度的上下文向量作为输入序列的抽象表示，解码器对抽象表示进行解码操作生成输出序列。

#### Processing Object：Series
序列在这里是个带有抽象意义的概念，指有顺序的一系列数据，可以是语音、文字，甚至可以是图片。它可以抽象地被表示为：
$$
(x_1, x_2, ..., x_n)，x_i∈R，i=1,2,...,n
$$


#### Model Structure：Encoder-Decoder
![alt text](C:\Users\14711\Desktop\Vision Transformer and Build from Scratch\image\image-1.png)[^2]
	Encoder和Decoder都分别是一个RNN。Encoder-Decoder中的***code***，就是把语言的发音、形式等东西剥离掉后，剩下的纯粹的语义关系[^3] 。

​										苹果---><img src="C:\Users\14711\AppData\Local\Temp\SGPicFaceTpBq\35508\07C73331.png" alt="07C73331" style="zoom:100%;" /><--- apple

两个完全不同的语言符号，需要通过一个实体把语义对应起来，建立语义联系[^3]。

**Encoder**：把长度可变的输入序列，压缩映射到一个固定形状的上下文向量(c)。
$$
ht = f(x_t, h_{t-1}) \\
\textbf{C} = q(h_1,...,h_T)
$$
大致地，输入序列会逐个被输入Encoder，每输入一个元素，Encoder就会更新hidden state，researcher把这种更新理解为Encoder“积累”了这个输入元素的信息，那么整个序列输入完后，Encoder的hidden state就会包含整个序列的信息。

当序列的最后一个元素被输入Encoder后，这时Encoder的hidden state，会被传递给Decoder，这时的hidden state可以被视为上下文向量。

**Decoder**：把固定形状的上下文向量逐步映射到长度可变的目标序列。

![alt text](C:\Users\14711\Desktop\Vision Transformer and Build from Scratch\image\image-3.png)
	Decoder的任务是逐步生成output，但它在生成的时候，输入有两部分：一部分是encoder的输出，即**上下文向量**，另一部分是 **Decoder之前生成了的元素**，

研究者们对Seq2Seq的希望是：它能够整合上下文信息，并结合整个序列的语义来完成seq2seq任务。实际上，这也是符合常识的：我们在翻译英语句子的时候，既要考虑每个单词的具体意思，也要考虑上下文构建出来的整体语境。

#### By the way: RNN

RNN(Recurrent Neural Network)，是一种用于处理**序列数据**的网络。它的结构相较于传统的前馈神经网络有所不同，主要体现在它的**循环连接**，使得网络能够保持对先前输入的记忆，并捕捉数据中的依赖关系（指序列中各个元素之间的相互联系。当模型能够利用前后的元素来帮助预测或理解当前的元素，达成较好效果时，称模型**较好地捕捉到了数据中的依赖关系**）

**输入**：对于一个给定时间步 t，RNN 接收当前的输入 $x_t$，表示当前时刻的特征

**隐藏状态**：RNN 的核心是**隐藏状态** $h_t$，它是对当前和先前输入的记忆。隐藏状态的更新取决于当前输入$x_t$和【上一个时间步的隐藏状态$h_{t-1}$

**输出**：RNN 还会输出一个 yty_tyt，这个输出是由当前隐藏状态计算得来的

在每一个时间步t，RNN的计算过程可以表示为:
$$
h_t = \sigma(W_hx_t + U_hh_{t-1} + b_h)
$$

- $W_h$：输入到hidden layer的权重矩阵
- $x_t$：当前时间步的输入
- $U_h$：上一时间步的隐藏状态到当前隐藏状态的权重矩阵
- $b_h$：隐藏层的bias
- $\sigma$：通常是激活函数，比如ReLu或者tanh

当输入经过所有隐藏层后，RNN会计算输出$y_t$：
$$
y_t = softmax(W_yh_t + b_y)
$$

- $W_y$：最后一个隐藏层到Output layer的权重矩阵
- $b_y$：Output layer的bias

RNN的特点：

- 权重共享：在所有时间步 t 中，隐藏层的权重 $W_h$和 $U_h$ 都是相同的，这使得 RNN 能够处理不同长度的序列
- 时间依赖：通过循环连接在网络中保留记忆，捕捉输入序列中的时间依赖性。
- 长序列问题：由于梯度消失和梯度爆炸，RNN 在训练过程中很难保持远距离时间步之间的信息。

RNN的变种：

- LSTM、GRU
- bi-RNN：同时使用前向和后向的 RNN 网络结构，使得每个时间步的隐藏状态能够同时考虑未来和过去的信息，提供了对序列的前向和反向上下文信息的建模能力。
- RNN with Attention 【We will focus on it later】

***【Encoder-decoder与 bi-RNN】***

- Encoder 可以是bi-RNN，让上下文向量有更丰富的语义信息，但在现代的 Seq2Seq 模型中，尤其是在使用 **Transformer** 或带有 **注意力机制** 的 RNN 时，bi-RNN 的使用没那么常见了。
- decoder不能是bi-RNN，这是由decoder的自回归性质（每一步的输出会作为下个时间步的输入）、目标序列的顺序生成（生成模型的生成过程，在本质上要求遵循时间的线性性，即从过去、当前预测未来）决定的。
- 在现代的模型（如 **Transformer**）中，**自注意力机制**（Self-Attention）取代了传统的 RNN 或 bi-RNN，它不依赖于顺序的计算，能够在所有位置之间自由地捕捉依赖关系。

### Training of Seq2Seq

1. 数据预处理: tokenization + word embedding
2. 参数初始化: decoder、输出层的weight和bias；Encoder的隐藏层状态
3. 设置超参数：learning rate, batch number...
4. 模型运行，输出经softmax处理后得到一个概率分布
5. 交叉熵函数计算损失
6. 反向传播更新参数

### Metrics of Seq2Seq

BLEU（bilingual evaluation understudy），通过与真实的标签序列进行比较来评估预测序列。

定义为：
$$
exp(min(0,1-\frac{len_{label}}{len_{pred}}))\prod_{n=1}^kp_n^{1/2^n}
$$

- n-gram精确度
- Precision
$$
P_n = \frac{机器翻译中的n-gram匹配数}{机器翻译中的n-gram总数}
$$
- 惩罚项(Brevity Penalty)
$$
f(x) = \left\{
  \begin{array}{ll}
  1 & \text{if } \textbf{c} > r \\
  exp(1-\frac{r}{c}) & \text{if } c ≤ r
  \end{array}
\right.
$$
c是机器翻译生成的句子的长度 ,r是参考翻译的长度

### Applications of Seq2Seq

一切【**序列->序列**】的任务：

- 机器翻译
- 语音识别
- 聊天机器人

![image-20250118153729564](C:\Users\14711\AppData\Roaming\Typora\typora-user-images\image-20250118153729564.png)

### Evaluation of Seq2Seq

Seq2Seq是NLP的经典模型了，说它经典是指：

1.它出现得很早，在当时很有用

2.现在它的应用没那么多了，但以它为深入Transformer的起点还是很好的

- pros
  - **处理可变长度的序列**：能适应不同长度的输入数据
  &nbsp;
  - **End to End**: 从原始输入数据（如文本、语音等）到期望输出序列的直接映射，无需进行显式的特征提取或工程化步骤
  &nbsp;
  - **可扩展**：天然具有模块化特性
- cons
  
  - 上下文向量固定维度 -> **信息压缩**：输入序列的全部信息需要被编码成一个固定维度的上下文向量，这导致了信息压缩和信息损失的问题，尤其是细粒度细节的丢失
  &nbsp;
  - **短期记忆**限制：由于循环神经网络（RNN）的固有特性，Seq2Seq模型在处理长序列时存在短期记忆限制，难以有效捕获和传递长期依赖性。这限制了模型在处理具有长距离时间步依赖的序列时的性能。
  &nbsp;  
  - **Exposure Bias**：在Seq2Seq模型的训练过程中，经常采用“teacher forcing”策略，即在每个时间步提供真实的输出作为解码器的输入。然而，这种训练模式与模型在推理时的自回归生成模式存在不一致性，导致模型在测试时可能无法很好地适应其自身的错误输出，这种现象被称为暴露偏差。

***【关于 Teacher forcing】***
	考虑这样一个简单模型（恒等映射）：
	输入是A，要求输出仍然是A
	（额外一提，要让模型学到恒等映射，可能不是件简单的事）

用户输入：我喜欢自然语言处理

模型输出：我喜欢自动化处理

问题出现在：当decoder生成完“动”，要生成下个元素时，decoder就会考虑到，上个元素是 “动”，然后生成 “化”。其实，从“动”这里就已经错了，但模型意识不到这个问题，会“错上加错”，很可能导致后面全错。

那么decoder为什么会倾向于输出“自动”，而不是“自然”呢，或许是Encoder太简单了，间接导致上下文向量的语义不够强，无法概括整个句子的核心意思，又或许是训练数据里“自动”要比“自然”更多，又或者是整个模型的参数太少了，导致欠拟合，也可能是损失函数有问题。可能的原因有太多了。

在训练过程中，为了训练效果（至少别越训越离谱，那样的话模型连收敛都收敛不了），就把decoder每一步接收的**decoder之前已经生成了的元素**，设置成了答案里的元素，比如输入还是 “我喜欢自然语言处理”，这时候哪怕 “动” 已经错了，下一步生成的时候，模型接收到的还是 “自然” 或者 “然”。

这样，模型在训练时，其实是有人类在监督着的，我们先天地保证了模型不会**偏离太远**，同时我们也就承担了模型没有获得**修正自身错误能力**的代价。

但在真实的应用中，模型只能靠自己，生成的每个词，都会依赖前面已经生成了的词。这时候，上述error propagation，或者说error accumulation，就会发生了。

### The Relationship between Seq2Seq and Encoder-decoder：模型 vs 框架

Seq2Seq模型是Encoder-decoder架构的一种具体应用

当我们说 Encoder-decoder



## Attention: is it ALL we need?[^4]

自经济学研究稀缺资源分配以来，人们正处在“注意力经济”时代， 即人类的注意力被视为可以交换的、有限的、有价值的且稀缺的商品。许多商业模式也被开发出来去利用这一点：在音乐或视频流媒体服务上，人们要么消耗注意力在广告上，要么付钱来隐藏广告； 为了在网络游戏世界的成长，人们要么消耗注意力在游戏战斗中， 从而帮助吸引新的玩家，要么付钱立即变得强大。 总之，注意力不是免费的。[^4]

注意力是稀缺的，而环境中干扰注意力的信息却不少。 人类的视觉神经系统大约每秒收到10^8^位的信息， 这远远超过了大脑能够处理的信息水平。 幸运的是，人类的祖先已经从经验中认识到 “并非感官的所有输入都是一样重要的”。 在整个人类历史中，这种只将注意力引向感兴趣的一小部分信息的能力， 使人类的大脑能够更明智地分配注意力资源来生存、发展。[^4]

人的视觉注意力机制有两种实现方式：

- 非自主性提示：视觉范围内的某个视觉对象具有突出特质。 【环境诱导注意力】

![img](https://b0.bdstatic.com/49ff69e23fec703a57d55d385f575b29.jpeg@h_1280)

- 自主性提示：人主动地将注意力聚集到某一处。 【后天训练得出经验-->注意力】

![image-20250118165026039](C:\Users\14711\AppData\Roaming\Typora\typora-user-images\image-20250118165026039.png)

对于非自主性提示实现的注意力机制，我们可以简单地用一个参数化的全连接层来模拟。

自然地，我们会关注如何建模自主性提示实现的注意力机制，这就要从具体的视觉注意行为中抽取出视觉注意过程，抽象为注意模型。为此，我们考察上述视觉注意力行为发生的过程：

人 去 注意 事物

人经过学习 知道哪一块是更重要的

如何判断这种"重要性"？

首先，这种重要是相对而言的，比较出来的，它必须对注意范围内的所有对象都进行作用，比较作用的大小；其次，重要性是有个目的的——人为什么会觉得这个东西重要，其背后必然存在着人的主观目的。

人依照着自己的目的，注意到了这个对象。

人拿着自己的"目的"，与注意范围内所有对象一一"反应"，最后得出注意对象。





所以，他们说：Attention is all you need.

但是，也不尽然。



![alt text](C:\Users\14711\Desktop\Vision Transformer and Build from Scratch\image\image-5.png)


## Decode Transformer: Prior Works

- 为什么layer normalization是放在每一层处理之后？
On Layer Normalization in the Transformer Architecture 2002.04745
- 为什么是layer normalization？别的行吗？
PowerNorm: Rethinking Batch Normalization in Transformers 2003.07845

## Comparison：Transformer & GPT & BERT





---

## References


[^1]:Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems (pp. 3104–3112).
[^2]:https://zh-v2.d2l.ai/chapter_recurrent-modern/seq2seq.html
[^3]:https://www.bilibili.com/video/BV1XH4y1T76e/?spm_id_from=333.1387.homepage.video_card.click&vd_source=747ba3a388b6a2ede61e61f1b864c1a6
[^4]:Mu Li, d2l: https://zh-v2.d2l.ai/chapter_attention-mechanisms/attention-cues.html



