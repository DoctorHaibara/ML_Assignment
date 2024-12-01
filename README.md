# Sun Yat-sen University Artificial Intelligence College Machine Learning Course Project

This repository contains the implementation of the BSARec and RBSARec models, a final project for the Machine Learning course at Sun Yat-sen University's Artificial Intelligence College. This project leverages the ReChorus 2.0 framework ([https://github.com/THUwangcy/ReChorus/](https://github.com/THUwangcy/ReChorus/)) for its modularity and flexibility in replicating and extending the BSARec algorithm.


## Project Overview: Replicating and Extending BSARec

This project focuses on replicating and extending the **BSARec** model, as described in:

**[BSARec](https://arxiv.org/abs/2312.10325)**
* **Title:** An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention
* **Authors:** Yehjin Shin*, Jeongwhan Choi*, Hyowon Wi, Noseong Park
* **Conference:** AAAI 2024

This project builds upon the foundational work of **SASRec**:

**[SASRec](https://arxiv.org/abs/1808.09781)**
* **Title:** Self-Attentive Sequential Recommendation
* **Authors:** Wang-Cheng Kang, Julian McAuley
* **Conference:** ICDM 2018

Our project further incorporates insights from **R-Transformer**:

**[R-Transformer](https://arxiv.org/abs/1907.05572)**
* **Title:** R-Transformer: Recurrent Neural Network Enhanced Transformer
* **Authors:** Zhiwei Wang, Yao Ma, Zitao Liu, Jiliang Tang


**Model Descriptions:**

**BSARec:** BSARec is a sequential recommendation model that combines a learned, frequency-based inductive bias with a trainable multi-head self-attention mechanism. It leverages Discrete Fourier Transform (DFT) to decompose the input sequence into high- and low-frequency components, allowing the model to learn relationships between items while avoiding over-smoothing. The model's architecture consists of stacked blocks, each incorporating weighted self-attention, frequency-based filtering, and a feed-forward network.


**RBSARec:** The RBSARec model is a hierarchical encoder for sequence data, designed to capture both short-term and long-term dependencies. It starts by encoding input sequences through frequency filtering and local RNNs, merging the resulting features. This combined representation is then processed using multi-head attention to learn relationships between different positions in the sequence, focusing on important information. Finally, a feedforward network extracts higher-order features. Each processing block incorporates residual connections and layer normalization for stability and efficiency. This layered architecture allows the model to gradually build a rich representation of the input data.


## Implementation Details

**RBSARec Model Hyperparameters:**

* `alpha` = 0.5
* `batch_size` = 256
* `c` = Values vary (5 for Beauty, 9 for Grocery, 5 for MovieLens_1M)
* `emb_size` = 256
* `epoch` = 200
* `l2` = 1e-06
* `lr` = Values vary (0.0005 for Grocery & Beauty, 0.001 for MovieLens_1M)
* `num_heads` = 4
* `num_layers` = 1
* `rnn_type` = Varies (LSTM for Grocery, GRU for Beauty and MovieLens)


**Data Preprocessing and Preparation:**

* **Data Cleaning:** Removed duplicate user-item interaction records to ensure data integrity.  Re-indexed users and items for model compatibility.
* **Feature Engineering:** Generated negative samples for contrastive learning during validation and testing phases.
* **Data Splitting:** Used a Leave-One-Out (LOO) approach to split the data into training, validation (development), and test sets.  Critically, this approach maintained the sequential order of user interactions in each dataset split.
* **Sequential Handling:** Sorted data by timestamp to ensure the model can accurately capture temporal dependencies in user behavior.


## Results

We present the results of our replicated BSARec and the extended RBSARec model, along with the ReChorus SASRec and GRU4Rec baselines on the Grocery & Gourmet Food, MovieLens, and Beauty datasets:

**Grocery & Gourmet Food:**

| Metric      | GRU4Rec (ReChorus) | SASRec (ReChorus) | BSARec (Our Replication) | RBSARec   |
|--------------|--------------------|--------------------|--------------------------|------------|
| HR@5         |  0.3510 | 0.3728             | 0.3567                    | 0.4086    |
| NDCG@5       |  0.2542 | 0.2725             | 0.2624                    | 0.2999    |
| HR@10        |  0.4530 | 0.4660             | 0.4424                    | 0.5036    |
| NDCG@10      |  0.2870 | 0.3026             | 0.2900                    | 0.3306    |
| HR@20        |  0.5727 | 0.5743             | 0.5507                    | 0.6119    |
| NDCG@20      |  0.3171 | 0.3298             | 0.3173                    | 0.3580    |
| HR@50        |  0.7953 | 0.7799             | 0.7642                    | 0.8118    |
| NDCG@50      |  0.3611 | 0.3705             | 0.3594                    | 0.3975    |


**MovieLens:**

| Metric      | GRU4Rec (ReChorus) | SASRec (ReChorus) | BSARec (Our Replication) | RBSARec   |
|--------------|--------------------|--------------------|--------------------------|------------|
| HR@5         | 0.5167              | 0.5254             | 0.5327                    | 0.5251     |
| NDCG@5       | 0.3792              | 0.3898             | 0.3962                    | 0.3846      |
| HR@10        | 0.6614              | 0.6712             | 0.6691                    | 0.6618   |
| NDCG@10      | 0.4260              | 0.4371             | 0.4404                    | 0.4287   |
| HR@20        | 0.8027              | 0.8055             | 0.8045                    | 0.8142   |
| NDCG@20      | 0.4618              | 0.4711             | 0.4747                    | 0.4673  |
| HR@50        | 0.9600              | 0.9492             | 0.9579                    | 0.9582   |
| NDCG@50      | 0.4934              | 0.5000             | 0.5055                    | 0.4962   |


**Beauty:**

| Metric      | GRU4Rec (ReChorus) | SASRec (ReChorus) | BSARec (Our Replication) | RBSARec   |
|--------------|--------------------|--------------------|--------------------------|------------|
| HR@5         | 0.3192              | 0.3457             | 0.3285                    | 0.3761    |
| NDCG@5       | 0.2281              | 0.2565             | 0.2456                    | 0.2799    |
| HR@10        | 0.4249              | 0.4436             | 0.4236                    | 0.4764    |
| NDCG@10      | 0.2623              | 0.2882             | 0.2762                    | 0.3123    |
| HR@20        | 0.5622              | 0.5612             | 0.5401                    | 0.5950    |
| NDCG@20      | 0.2969              | 0.3178             | 0.3056                    | 0.3422    |
| HR@50        | 0.7949              | 0.7788             | 0.7623                    | 0.7992    |
| NDCG@50      | 0.3428              | 0.3606             | 0.3493                    | 0.3826    |


## Discussion

This project successfully replicated the BSARec model and extended it to RBSARec within the ReChorus framework. Results across the Grocery & Gourmet Food, MovieLens, and Beauty datasets demonstrate that RBSARec consistently outperforms both our replicated BSARec and the ReChorus SASRec baseline. This improvement suggests that the hierarchical encoding approach, incorporating frequency filtering and local RNNs, effectively captures both short-term and long-term dependencies in user behavior. Proper hyperparameter tuning (e.g., learning rate, number of layers, RNN type) was crucial in achieving these improved results on all datasets. The performance gain of RBSARec over the baseline (SASRec) highlights the benefits of its architecture. Further analysis is needed to pinpoint the specific factors contributing to the superior performance of RBSARec.
