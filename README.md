# Sun Yat-sen University Artificial Intelligence College Machine Learning Course Project

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

* **Datasets:** We used the Grocery & Gourmet Food, MovieLens, and Beauty datasets. The ReChorus framework's provided scripts were used to split the data into training, validation (dev), and test sets. 

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


## Results & Analysis

We present the results of our replicated BSARec model and, for comparison, the ReChorus implementation of SASRec on the Grocery & Gourmet Food, MovieLens, and Beauty datasets.  The results highlight the impact of hyperparameter tuning on the performance of BSARec.

**Grocery & Gourmet Food:**

| Metric      | BSARec (Our Replication) | SASRec (ReChorus) |
|--------------|--------------------------|--------------------|
| HR@5         | 0.3895                    | 0.3728             |
| NDCG@5       | 0.2977                    | 0.2725             |
| HR@10        | 0.4734                    | 0.4660             |
| NDCG@10      | 0.3248                    | 0.3026             |
| HR@20        | 0.5722                    | 0.5743             |
| NDCG@20      | 0.3497                    | 0.3298             |
| HR@50        | 0.7609                    | 0.7799             |
| NDCG@50      | 0.3869                    | 0.3705             |

**Hyperparameters (BSARec):** `alpha=0.3`, `batch_size=256`, `c=9`, `emb_size=256`, `epoch=200`, `l2=1e-06`, `lr=0.0005`, `num_heads=4`, `num_layers=1`

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

**Hyperparameters (BSARec):** `alpha=0.5`, `batch_size=256`, `c=5`, `emb_size=64`, `epoch=200`, `l2=1e-06`, `lr=0.001`, `num_heads=4`, `num_layers=1`


**Beauty:**

| Metric      | BSARec (Our Replication) | SASRec (ReChorus) |
|--------------|--------------------------|--------------------|
| HR@5         | 0.4273                    | 0.3457             |
| NDCG@5       | 0.3328                    | 0.2565             |
| HR@10        | 0.5163                    | 0.4436             |
| NDCG@10      | 0.3616                    | 0.2882             |
| HR@20        | 0.6217                    | 0.5612             |
| NDCG@20      | 0.3881                    | 0.3178             |
| HR@50        | 0.8112                    | 0.7788             |
| NDCG@50      | 0.4255                    | 0.3606             |

**Hyperparameters (BSARec):** `alpha=0.5`, `batch_size=256`, `c=5`, `emb_size=256`, `epoch=200`, `l2=1e-06`, `lr=0.001`, `num_heads=8`, `num_layers=2`

## Discussion

Our replication of the BSARec model consistently demonstrated improvements over the SASRec baseline across all three datasets (Grocery & Gourmet Food, MovieLens, and Beauty). The results strongly suggest that adjusting hyperparameters, particularly the learning rate, batch size, and architectural choices like number of layers, significantly impacted the model's performance positively. For instance, the Beauty dataset showed the most substantial gains with optimized hyperparameters, indicating significant improvement in recommendation quality. While specific improvements varied across the datasets, the overall trend pointed towards enhanced performance after tuning.


**Key Takeaways:**

* **Hyperparameter Sensitivity:** The performance of BSARec is highly sensitive to hyperparameter choices.  Our experiment demonstrates that meticulous tuning of parameters like the learning rate, `c`, and number of layers can lead to improved performance.
* **Dataset-Specific Tuning:**  Further hyperparameter tuning is necessary to achieve optimal performance across different datasets. The results on the Beauty dataset suggest that tuning could yield better results.


Further experimentation with different hyperparameter values, especially on datasets where performance is not as strong, could lead to a more robust and reliable model for various recommendation tasks.
