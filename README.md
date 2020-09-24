# Transformer Based Multi-Source Domain Adaptation
Dustin Wright and Isabelle Augenstein

To appear in EMNLP 2020. Read the preprint: https://arxiv.org/abs/2009.07806

<p align="center">
  <img src="multisource-domain-adaptation.png" alt="PUC">
</p>

In practical machine learning settings, the data on which a model must make predictions often come from a different distribution than the data it was trained on. Here, we investigate the problem of unsupervised multi-source domain adaptation, where a model is trained on labelled data from multiple source domains and must make predictions on a domain for which no labelled data has been seen. Prior work with CNNs and RNNs has demonstrated the benefit of mixture of experts, where the predictions of multiple domain expert classifiers are combined; as well as domain adversarial training, to induce a domain agnostic representation space. Inspired by this, we investigate how such methods can be effectively applied to large pretrained transformer models. We find that domain adversarial training has an effect on the learned representations of these models while having little effect on their performance, suggesting that large transformer-based models are already relatively robust across domains. Additionally, we show that mixture of experts leads to significant performance improvements by comparing several variants of mixing functions, including one novel mixture based on attention. Finally, we demonstrate that the predictions of large pretrained transformer based domain experts are highly homogenous, making it challenging to learn effective functions for mixing their predictions.

# Citing

```bib
@inproceedings{wright2020transformer,
  title={{Transformer Based Multi-Source Domain Adaptation}},
  author={Dustin Wright and Isabelle Augenstein},
  booktitle = {Proceedings of EMNLP},
  publisher = {Association for Computational Linguistics},
  year = 2020
}
```

# Recreating Results

The code is currently being prepared and will be released shortly.
