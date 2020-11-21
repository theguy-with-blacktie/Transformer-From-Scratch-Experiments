<b>Why Heads in Self-Attention?</b><br>
Consider the following example:
<i>mary, gave, roses, to, susan</i>. We see the word 'gave' has different relations to different parts of the sentence. 'mary' expresses who's doing the giving, 'roses' expresses what's being given, and 'susan' expresses who the recipient is.
<br>
In a single self-attention operation, all this information just gets summed together. If Susan gave Mary the roses instead, the output vector 𝐲gave would be the same, even though the meaning has changed.
We think of <i>h</i> attention heads as <i>h</i> separate sets of three weight matrices of queries, keys & values, but it would be more efficient to combine all the 
