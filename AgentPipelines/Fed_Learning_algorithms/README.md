
## Methods 🧬



<!-- <details> -->
<summary><b>Traditional FL Methods</b></summary>

- ***FedAvg*** -- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (AISTATS'17)
- ***FedAvgM*** -- [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) (ArXiv'19)
- ***FedProx*** -- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) (MLSys'20)
- ***SCAFFOLD*** -- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) (ICML'20)
- ***MOON*** -- [Model-Contrastive Federated Learning](http://arxiv.org/abs/2103.16257) (CVPR'21)
- ***FedDyn*** -- [Federated Learning Based on Dynamic Regularization](http://arxiv.org/abs/2111.04263) (ICLR'21)
- ***FedLC*** -- [Federated Learning with Label Distribution Skew via Logits Calibration](http://arxiv.org/abs/2209.00189) (ICML'22)
- ***FedGen*** -- [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](https://arxiv.org/abs/2105.10056) (ICML'21)
- ***CCVR*** -- [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data](https://arxiv.org/abs/2106.05001) (NeurIPS'21)
- ***FedOpt*** -- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) (ICLR'21)
- ***Elastic Aggregation*** -- [Elastic Aggregation for Federated Optimization](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) (CVPR'23)
- ***FedFed*** -- [FedFed: Feature Distillation against Data Heterogeneity in Federated Learning](http://arxiv.org/abs/2310.05077) (NeurIPS'23)

<!-- </details> -->

<!-- <details> -->
<summary><b>Personalized FL Methods</b></summary>

- ***pFedSim*** -- [pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning](https://arxiv.org/abs/2305.15706) (ArXiv'23)
- ***Local-Only*** -- Local training only (without communication).
- ***FedMD*** -- [FedMD: Heterogenous Federated Learning via Model Distillation](http://arxiv.org/abs/1910.03581) (NeurIPS'19)
- ***APFL*** -- [Adaptive Personalized Federated Learning](http://arxiv.org/abs/2003.13461) (ArXiv'20)
- ***LG-FedAvg*** -- [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) (ArXiv'20)
- ***FedBN*** -- [FedBN: Federated Learning On Non-IID Features Via Local Batch Normalization](http://arxiv.org/abs/2102.07623) (ICLR'21)
- ***FedPer*** -- [Federated Learning with Personalization Layers](http://arxiv.org/abs/1912.00818) (AISTATS'20)
- ***FedRep*** -- [Exploiting Shared Representations for Personalized Federated Learning](http://arxiv.org/abs/2102.07078) (ICML'21)
- ***Per-FedAvg*** -- [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9a9203a44cdad-Abstract.html) (NeurIPS'20)
- ***pFedMe*** -- [Personalized Federated Learning with Moreau Envelopes](http://arxiv.org/abs/2006.08848) (NeurIPS'20)
- ***FedEM*** -- [Federated Multi-Task Learning under a Mixture of Distributions](https://arxiv.org/abs/2108.10252) (NIPS'21)
- ***Ditto*** -- [Ditto: Fair and Robust Federated Learning Through Personalization](http://arxiv.org/abs/2012.04221) (ICML'21)
- ***pFedHN*** -- [Personalized Federated Learning using Hypernetworks](http://arxiv.org/abs/2103.04628) (ICML'21)
- ***pFedLA*** -- [Layer-Wised Model Aggregation for Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Layer-Wised_Mdel_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.html) (CVPR'22)
- ***CFL*** -- [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991) (ArXiv'19)
- ***FedFomo*** -- [Personalized Federated Learning with First Order Model Optimization](http://arxiv.org/abs/2012.08565) (ICLR'21)
- ***FedBabu*** -- [FedBabu: Towards Enhanced Representation for Federated Image Classification](https://arxiv.org/abs/2106.06042) (ICLR'22)
- ***FedAP*** -- [Personalized Federated Learning with Adaptive Batchnorm for Healthcare](https://arxiv.org/abs/2112.00734) (IEEE'22)
- ***kNN-Per*** -- [Personalized Federated Learning through Local Memorization](http://arxiv.org/abs/2111.09360) (ICML'22)
- ***MetaFed*** -- [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](http://arxiv.org/abs/2206.08516) (IJCAI'22)
- ***FedRoD*** -- [On Bridging Generic and Personalized Federated Learning for Image Classification](https://arxiv.org/abs/2107.00778) (ICLR'22)
- ***FedProto*** -- [FedProto: Federated prototype learning across heterogeneous clients](https://arxiv.org/abs/2105.00243) (AAAI'22)
- ***FedPAC*** -- [Personalized Federated Learning with Feature Alignment and Classifier Collaboration](https://arxiv.org/abs/2306.11867v1) (ICLR'23)
- ***FedALA*** -- [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://arxiv.org/abs/2212.01197) (AAAI'23)
- ***PeFLL*** -- [PeFLL: Personalized Federated Learning by Learning to Learn](https://openreview.net/forum?id=MrYiwlDRQO) (ICLR'24)
- ***FLUTE*** -- [Federated Representation Learning in the Under-Parameterized Regime](https://openreview.net/forum?id=LIQYhV45D4) (ICML'24)
- ***FedAS*** -- [FedAS: Bridging Inconsistency in Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.html) (CVPR'24)
- ***pFedFDA*** -- [pFedFDA: Personalized Federated Learning via Feature Distribution Adaptation](http://arxiv.org/abs/2411.00329) (NeurIPS 2024)
- ***Floco*** -- [Federated Learning over Connected Modes](https://openreview.net/forum?id=JL2eMCfDW8) (NeurIPS'24)
<!-- </details> -->


<!-- <details> -->
<summary><b>FL Domain Generalization Methods</b></summary>

- ***FedSR*** -- [FedSR: A Simple and Effective Domain Generalization Method for Federated Learning](https://openreview.net/forum?id=mrt90D00aQX) (NeurIPS'22)
- ***ADCOL*** -- [Adversarial Collaborative Learning on Non-IID Features](https://proceedings.mlr.press/v202/li23j.html) (ICML'23)
- ***FedIIR*** -- [Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships](https://openreview.net/pdf?id=JC05k0E2EM) (ICML'23)
<!-- </details> -->


### Implementing FL Method

The `package()` at server-side class is used for assembling all parameters server need to send to clients. Similarly, `package()` at client-side class is for parameters clients need to send back to server. You should always has `super().package()` in your override implementation. 

- Consider to inherit your method classes from [`FedAvgServer`](src/server/fedavg.py) and [`FedAvgClient`](src/client/fedavg.py) for maximum utilizing FL-bench's workflow.

- You can also inherit your method classes from advanced methods, e.g., FedBN, FedProx, etc. Which will inherit all functions, variables and hyperparamter settings. If you do that, you need to careful design your method in order to avoid potential hyperparamters and workflow conflicts.
```python
class YourServer(FedBNServer):
  ...

class YourClient(FedBNClient):
  ...
```


- For customizing your server-side process, consider to override the `package()` and `aggregate_client_updates()`.

- For customizing your client-side training, consider to override the `fit()`, `set_parameters()` and `package()`.

You can find all details in [`FedAvgClient`](src/client/fedavg.py) and [`FedAvgServer`](src/server/fedavg.py), which are the bases of all implementations in FL-bench.

```
