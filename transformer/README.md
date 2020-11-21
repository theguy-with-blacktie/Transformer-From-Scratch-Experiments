# NOTES [SelfAttention & Transformer]

## Self Attention (Attention is All You Need!)

### Self Attention in only Two Matrix Multiplications
Self-attention is a sequence-to-sequence operation: a sequence of vectors goes in, and a sequence of vectors comes out. Let's call the input vectors <i><b>x<sub>1</sub>,</b></i><i><b>x<sub>2</sub>,..,</b></i><i><b>x<sub>t</sub>,</b></i> and the corresponding output vectors <i><b>y<sub>1</sub>,</b></i><i><b>y<sub>2</sub>,..,</b></i><i><b>y<sub>t</sub>,</b></i>. The vectors all have dimension <i><b>k</b></i>.<br>
To produce output vector <i><b>y<sub>i</sub>,</b></i> the self attention operation simply takes a <i>weighted average over all the input vectors</i><br>
<img src="https://latex.codecogs.com/svg.latex?y_{i}=\sum&space;_{j}w_{ij}x_{i}" title="y_{i}=\sum _{j}w_{ij}x_{i}" />
<b>Why Heads in Self-Attention?</b><br>
Consider the following example:
<i>mary, gave, roses, to, susan</i>. We see the word 'gave' has different relations to different parts of the sentence. 'mary' expresses who's doing the giving, 'roses' expresses what's being given, and 'susan' expresses who the recipient is.
<br>
In a single self-attention operation, all this information just gets summed together. If Susan gave Mary the roses instead, the output vector 𝐲gave would be the same, even though the meaning has changed.
We think of <i>h</i> attention heads as <i>h</i> separate sets of three weight matrices of queries, keys & values, but it would be more efficient to combine all the 
