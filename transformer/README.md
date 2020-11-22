# NOTES [SelfAttention & Transformer]

## Self Attention (Attention is All You Need!)

### Self Attention in only Two Matrix Multiplications
Self-attention is a sequence-to-sequence operation: a sequence of vectors goes in, and a sequence of vectors comes out. Let's call the input vectors <i><b>x<sub>1</sub>,</b></i><i><b>x<sub>2</sub>,..,</b></i><i><b>x<sub>t</sub>,</b></i> and the corresponding output vectors <i><b>y<sub>1</sub>,</b></i><i><b>y<sub>2</sub>,..,</b></i><i><b>y<sub>t</sub>,</b></i>. The vectors all have dimension <i><b>k</b></i>.<br>
To produce output vector <i><b>y<sub>i</sub>,</b></i> the self attention operation simply takes a <i>weighted average over all the input vectors</i><br>
<div style="text-align:center"><img src="https://latex.codecogs.com/svg.latex?y_{i}=\sum&space;_{j}w_{ij}x_{i}" title="y_{i}=\sum _{j}w_{ij}x_{i}" /></div>
Where <i><b>j</b></i> indexes over the whole sequence and the weights sum to one over all <i><b>j</b></i>. The weight <i><b>w<sub>ij</sub></b></i> is not a parameter, as in a normal neural net, but it is derived from a function over <i><b>x<sub>i</sub></b></i> and <i><b>x<sub>j</sub></b></i>. The simplest option for this function is the dot product:<br>
<div style="text-align:center">
<img src="https://latex.codecogs.com/svg.latex?w^{'}_{ij}=x^{T}_{ij}x_{j}" title="w^{'}_{ij}=x^{T}_{ij}x_{j}" /><br>
<i>Note that <b>x<sub>i</sub></b></i> is the input vector at the same position as the current output vector <b>y<sub>i</sub></b>. For the next output vector, we get an entirely new series of dot products, and a different weighted sum.</i>
</div>
<br>
The dot product gives us a value anywhere between negative and positive infinity, so we apply a softmax to map the values to [0,1] and to ensure that they sum to 1 over the whole sequence:
<div style="text-align:center">
<img src="https://latex.codecogs.com/svg.latex?w_{ij}=\frac{\sigma&space;(w^{'}_{ij})}{\sum&space;_{j}\sigma(w^{'}_{ij})}" title="w_{ij}=\frac{\sigma (w^{'}_{ij})}{\sum _{j}\sigma(w^{'}_{ij})}" />
</div>
<b>And that's the basic operation of self attention.</b>

#### Implementation
The first thing we should do is work out how to express the self attention in matrix multiplications. A naive implementation that loops all vectors to compute the weights and outputs would be too much slow.<br>
We'll represent the input, a sequence of <i><b>t</b></i> vectors of dimension <i><b>k</b></i> as a <i><b>t x k</b></i> matrix <i><b>X</b></i>. Including a minibatch dimension <i><b>b</b></i>, gives us an input tensor of size <i><b>(b, t, k)</b></i>.<br>
The set of all raw dot products <b><i>w<sup>'</sup><sub>ij</sub></i></b> forms a matrix, which we can compute simply by multiplying <b>X</b> by its transpose:<br>
```python
import torch<br>
import torch.nn.functional as F
# assume we have some tensor x with size (b,t,k)
x = ...
raw_weights = torch.bnm(x,x.transpose(1,2))
```
Then, to turn the rawweights <b><i>w<sup>'</sup><sub>ij</sub></i></b> into positive values that sum to one, we apply a <i>row-wise</i> softmax:<br>
```python
weights = F.softmax(raw_weights, dim=2)
```
<br>
Finally, to compute the output sequence, we just multiply the weight matrix by <b>X</b>. This results in a batch of output matrices <b>Y</b> of size (b, y, k) whose rows are weighted sums over the rows of <b>X</b>.
<br>

```python
y = torch.bnm(weights, x)
```
<br>

#### Additional Tricks
The actual self-attention used in modern transformers relies on three additional tricks.<br>
1. Queries, keys and values
Every input vector <b><i>x<sub>i</sub></i></b> is used in three different ways in the self attention operation:
* It is compared to every other vector to establish the weights for its own output <b><i>y<sub>i</sub></i></b>.
* It is compared to ever other vector to establish the weights for the output of the j-th vector <b><i>y<sub>j</sub></i></b>.
* It is used as part of the weighted sum to compute each output vector once the weights have been established.

These roles are often called the <b>query</b>, the <b>key</b> and the <b>value</b>.
<br>
Below figure will provide you more insight on how actually the input in used via <b>query</b>, <b>key</b> and the <b>value</b> matrices.
![Query Key Value Figure](https://github.com/theguy-with-blacktie/Transformer-From-Scratch-Experiments/blob/master/transformer/qkv.PNG?raw=true)
<br>

2. Scaling the dot product
The softmax function can be very sensitive to very large input values. These kill the gradient, and slow down learning, or cause it to stop altogether. Since the average value of the dot product grows with the embedding dimension <i>k</i>, it helps to scale the dot product back a little to stop the inputs to the softmax function from growing too large:<br>
<p style="text-align:centre">
<img src="https://latex.codecogs.com/svg.latex?w^{`}_{ij}=\frac{Q^{T}K}{\sqrt{k}}" title="w^{`}_{ij}=\frac{Q^{T}K}{\sqrt{k}}" />
</p>
<b>Why Heads in Self-Attention?</b><br>
Consider the following example:
<i>mary, gave, roses, to, susan</i>. We see the word 'gave' has different relations to different parts of the sentence. 'mary' expresses who's doing the giving, 'roses' expresses what's being given, and 'susan' expresses who the recipient is.
<br>
In a single self-attention operation, all this information just gets summed together. If Susan gave Mary the roses instead, the output vector 𝐲gave would be the same, even though the meaning has changed.
We think of <i>h</i> attention heads as <i>h</i> separate sets of three weight matrices of queries, keys & values, but it would be more efficient to combine all the 
