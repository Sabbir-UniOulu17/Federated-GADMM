# Distributed Machine Learning

The file named f_gadmm.ipynb is the primary implementation of one of the proposed distributed machine learning algorithms called Federated Group Alternating Direct Method of Multipliers (F-GADMM).CNN with five layers model has been considered here. The network architecture, number of users and iterations can be adjusted according to the requirements but it should be bear in mind that more users and iterations mean longer time taken to generate the results.

The file named q_fgadmm.py is the quantized version of F-GADMM, where we primarily implemented with a very simple MLP neural networks with four users.


#Citation

@article{elgabli2019fgadmm,
  title={L-FGADMM: Layer-Wise Federated Group ADMM for Communication Efficient Decentralized Deep Learning},
  author={Elgabli, Anis and Park, Jihong and Ahmed, Sabbir and Bennis, Mehdi},
  journal={arXiv preprint arXiv:1911.03654},
  year={2019}
}
