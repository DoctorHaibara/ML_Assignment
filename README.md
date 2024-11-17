# Replicating the BSARec Paper using ReChorus: Sun Yat-sen University Artificial Intelligence College Machine Learning Course Project

This repository contains the implementation of the BSARec model, a final project for the Machine Learning course at Sun Yat-sen University's Artificial Intelligence College. This project leverages the ReChorus 2.0 framework ([https://github.com/THUwangcy/ReChorus/](https://github.com/THUwangcy/ReChorus/)) for its modularity and flexibility in replicating the BSARec algorithm.


## Project Overview: Replicating BSARec

This project focuses on replicating the **BSARec** model, as described in:

**[BSARec](https://arxiv.org/abs/2312.10325)**
* **Title:** An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention
* **Authors:** Yehjin Shin*, Jeongwhan Choi*, Hyowon Wi, Noseong Park
* **Conference:** AAAI 2024


This project builds upon the foundational work of **SASRec**:

**[SASRec](https://arxiv.org/abs/1808.09781)**
* **Title:** Self-Attentive Sequential Recommendation
* **Authors:** Wang-Cheng Kang, Julian McAuley
* **Conference:** ICDM 2018


## Implementation Details

This project utilizes the **ReChorus 2.0** framework ([https://github.com/THUwangcy/ReChorus/](https://github.com/THUwangcy/ReChorus/)) for implementing the BSARec model. ReChorus provided a structured and efficient environment for development, training, and evaluation.  Its modular design facilitated the integration of the BSARec components.

**Specific Implementation Choices:**

* **Datasets:** We used the Grocery & Gourmet Food and MovieLens datasets. The ReChorus framework's provided scripts were used to split the data into training, validation (dev), and test sets. 

* **Model Architecture:** The BSARec model architecture was implemented as follows:

```
INFO:root:BSARec(
  (i_embeddings): Embedding(8714, 64)
  (p_embeddings): Embedding(21, 64)
  (BSARecBlock): ModuleList(
    (0): BSARecBlock(
      (layer): BSARecLayer(
        (filter_layer): FrequencyLayer(
          (out_dropout): Dropout(p=0, inplace=False)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
        )

        (attention_layer): MultiHeadAttention(
          (q_linear): Linear(in_features=64, out_features=64, bias=True)
          (k_linear): Linear(in_features=64, out_features=64, bias=True)
          (v_linear): Linear(in_features=64, out_features=64, bias=True)
        )
      )
      (layer_norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0, inplace=False)
      (linear1): Linear(in_features=64, out_features=64, bias=True)
      (linear2): Linear(in_features=64, out_features=64, bias=True)
      (layer_norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (dropout2): Dropout(p=0, inplace=False)
    )
  )
)
```

Key architectural parameters include:

* **Embedding Dimension:** 64
* **Number of Attention Heads:** 4
* **Number of Layers (BSARecBlocks):** 1
* **Activation Function:** GeLU
* **Dropout Rate:** 0


* **Training:** We trained the model using the Adam optimizer with a learning rate of 0.001 and a batch size of 256.  A total of 200 epochs were used.  No explicit hyperparameter tuning was performed beyond those specified in the configuration file.  L2 regularization was applied with a weight decay of 1e-06.  Early stopping was implemented with a patience of 10 epochs.


## Results

We present the results of our replicated BSARec model and, for comparison, the ReChorus implementation of SASRec on both datasets:

**Grocery & Gourmet Food:**

| Metric      | BSARec (Our Replication) | SASRec (ReChorus) |
|--------------|--------------------------|--------------------|
| HR@5         | 0.3567                    | 0.3728             |
| NDCG@5       | 0.2624                    | 0.2725             |
| HR@10        | 0.4424                    | 0.4660             |
| NDCG@10      | 0.2900                    | 0.3026             |
| HR@20        | 0.5507                    | 0.5743             |
| NDCG@20      | 0.3173                    | 0.3298             |
| HR@50        | 0.7642                    | 0.7799             |
| NDCG@50      | 0.3594                    | 0.3705             |


**MovieLens:**

| Metric      | BSARec (Our Replication) | SASRec (ReChorus) |
|--------------|--------------------------|--------------------|
| HR@5         | 0.5327                    | 0.5254             |
| NDCG@5       | 0.3962                    | 0.3898             |
| HR@10        | 0.6691                    | 0.6712             |
| NDCG@10      | 0.4404                    | 0.4371             |
| HR@20        | 0.8045                    | 0.8055             |
| NDCG@20      | 0.4747                    | 0.4711             |
| HR@50        | 0.9579                    | 0.9492             |
| NDCG@50      | 0.5055                    | 0.5000             |


## Discussion

Our replication of the BSARec model, implemented within the ReChorus framework, demonstrates performance comparable to the ReChorus baseline implementation of SASRec across both the Grocery & Gourmet Food and MovieLens datasets. This indicates a successful replication of the BSARec algorithm, achieving similar levels of HR@K and NDCG@K scores as the established ReChorus baseline.
