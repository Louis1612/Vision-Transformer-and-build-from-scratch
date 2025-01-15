[TOC]
#Datawhale组队学习
# Vision Transformer and build from scratch

## Seq2Seq

### Why we begin from Seq2Seq?
Transformer模型在Seq2Seq的基础上进行了重要的改进——Transformer抛弃了RNN和LSTM的结构，改为使用自注意力机制（Self-Attention），使得模型能够并行处理序列并捕捉长距离的依赖关系。从这里开始，可以在比较中慢慢掌握更复杂的模型设计。
### What is Seq2Seq
![alt text](image-4.png)
#### Origin：
2014年，机器翻译任务 (Sutskever et al., 2014)[^1]中，为了解决输入与输出序列长短未知的问题，研究者引入了Seq2Seq模型
#### A Quick Flight
编码器将输入序列压缩为一个固定维度的上下文向量作为输入序列的抽象表示，解码器对抽象表示进行解码操作生成输出序列。
#### Processing Object：Series
序列在这里是个带有抽象意义的概念，指有顺序的一系列数据，可以是语音、文字，甚至可以是图片。它可以抽象地被表示为：
$$
(x_1, x_2, ..., x_n)，x_i∈R，i=1,2,...,n
$$


#### Model Structure：Encoder-decoder
![alt text](image-1.png)[^2]
Encoder和decoder都分别是一个RNN
Encoder：生成包含输入序列的全部重要信息的上下文向量(c)。
$$
ht = f(x_t, h_{t-1}) \\
\textbf{C} = q(h_1,...,h_T)
$$
大致地，输入序列会逐个被输入Encoder，每输入一个元素，Encoder就会更新hidden state，researcher把这种更新理解为Encoder“积累”了这个输入元素的信息，那么整个序列输入完后，Encoder的hidden state就会包含整个序列的信息。

当序列的最后一个元素被输入Encoder后，这时Encoder的hidden state，会被传递给decoder，这时的hidden state可以被视为上下文向量。

decoder：逐步生成目标序列（输出序列）的每个元素。

![alt text](image-3.png)
decoder的任务是逐步生成output，但它在生成的时候，输入有两部分：一部分是encoder的输出，即**上下文向量**，另一部分是**decoder之前生成了的元素**，

研究者们对Seq2Seq的希望是：它能够整合上下文信息，并结合整个序列的语义来完成seq2seq任务。实际上，这也是符合常识的：我们在翻译英语句子的时候，既要考虑每个单词的具体意思，也要考虑上下文构建出来的整体语境。
#### Training Process
1. 数据预处理: tokenization + word embedding
2. 参数初始化: decoder、输出层的weight和bias；Encoder的隐藏层状态
3. 设置超参数：learning rate, batch number...
4. 模型运行，输出经softmax处理后得到一个概率分布
5. 交叉熵函数计算损失
6. 反向传播更新参数

#### Metrics
BLEU（bilingual evaluation understudy）
通过与真实的标签序列进行比较来评估预测序列
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
#### Applications
机器翻译
语音识别
聊天机器人
一切【**序列->序列**】的任务
#### The Evaluation of Seq2Seq
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
  
  - 上下文向量**信息压缩**：输入序列的全部信息需要被编码成一个固定维度的上下文向量，这导致了信息压缩和信息损失的问题，尤其是细粒度细节的丢失
&nbsp;
  - **短期记忆**限制：由于循环神经网络（RNN）的固有特性，Seq2Seq模型在处理长序列时存在短期记忆限制，难以有效捕获和传递长期依赖性。这限制了模型在处理具有长距离时间步依赖的序列时的性能。
&nbsp;  
  - **Exposure Bias**：在Seq2Seq模型的训练过程中，经常采用“teacher forcing”策略，即在每个时间步提供真实的输出作为解码器的输入。然而，这种训练模式与模型在推理时的自回归生成模式存在不一致性，导致模型在测试时可能无法很好地适应其自身的错误输出，这种现象被称为暴露偏差。

==**关于Teacher forcing**==
考虑这样一个简单模型（恒等映射）：
输入是A，要求输出仍然是A
（额外一提，要让模型学到恒等映射，可能不是件简单的事）

用户输入：我喜欢自然语言处理

模型输出：我喜欢自动化处理

问题出现在：当decoder生成完“动”，要生成下个元素时，decoder就会考虑到，上个元素是 “动”，然后生成 “化”。其实，从“动”这里就已经错了，但模型意识不到这个问题，会“错上加错”，很可能导致后面全错。

那么decoder为什么会倾向于输出“自动”，而不是“自然”呢，或许是Encoder太简单了，间接导致上下文向量的语义不够强，无法概括整个句子的核心意思，又或许是训练数据里“自动”要比“自然”更多，又或者是整个模型的参数太少了，导致欠拟合，也可能是损失函数有问题。可能的原因有太多了。

在训练过程中，为了训练效果（至少别越训越离谱，那样的话模型连收敛都收敛不了），就把decoder每一步接收的**decoder之前已经生成了的元素**，设置成了答案里的元素，比如输入还是 “我喜欢自然语言处理”，这时候哪怕 “动” 已经错了，下一步生成的时候，模型接收到的还是 “自然” 或者 “然”。

这样，模型在训练时，其实是有人类在监督着的，我们先天地保证了模型不会**偏离太远**，同时我们也就承担了模型没有获得**修正自身错误能力**的代价。

但在真实的应用中，模型只能靠自己，生成的每个词，都会依赖前面已经生成了的词。这时候，上述error propagation，或者说error accumulation，就会发生了

### The Relationship between Seq2Seq and Encoder-decoder
Seq2Seq模型是Encoder-Decoder架构的一种具体应用
Seq2Seq更强调**目的**，Encoder-Decoder 更强调**方法**

### Attention: is it ALL we need?
![alt text](image-5.png)


## Decode Transformer: Prior Works

- 为什么layer normalization是放在每一层处理之后？
On Layer Normalization in the Transformer Architecture 2002.04745
- 为什么是layer normalization？别的行吗？
PowerNorm: Rethinking Batch Normalization in Transformers 2003.07845


[^1]:Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems (pp. 3104–3112).
[^2]:https://zh-v2.d2l.ai/chapter_recurrent-modern/seq2seq.html