On each dataset, seven algorithms were implemented. AdaClip1 [1] and AdaClip2 [2] are two adaptive clipping methods. MAPA is our proposed adaptive method (including MAPA-S and MAPA-C). FixDP means using the fixed value to clip the gradients (including FixDP-S and FixDP-C). All the above algorithms are private where using the differential privacy to protect the gradients. NonDP means the asynchronous mode without considering privacy protection. 

In the above algorithms, AdaClip1, FixDP-S, MAPA-S are sample-level algorithms, while AdaClip2, FixDP-C, MAPA-C are client-level algorithms. 

The file names present the studied scenarios. For example, REDDIT_AdaClip1_ASyn_08_flat.py presents that we use AdaClip1 for Reddit dataset, considering the asynchronous mode with the flat clipping strategy.

# Remark
(1) We use the Gaussian mechanism for protecting the privacy of gradients or model parameters. The number in the file name is a noise scale z in Gaussian distribution N(0, (zS)^2), where S is the sensitivity in differential privacy. For example, in REDDIT_AdaClip1_ASyn_08_flat.py, z=0.8.
(2) All files can be extended to synchronous mode and per-layer clipping strategy. The corresponding codes have been included in the files.
(3) To run the files, the file dataloader.py has to be replaced by our provided new file. 
Please find the environment where PySyft installed and replace the dataloader.py under Lib\site-packages\syft\frameworks\torch\fl with the dataloader.py we give.
(4) The used datasets are not uploaded considering its size. The used datasets need to be downloaded from http://leaf.cmu.edu or https://pan.baidu.com/s/1azWga0I-KUpb4DbBi65x-A (with extraction code ynz3). 


[1] Pichapati V, Suresh A T, Yu F X, et al. AdaCliP: Adaptive clipping for private SGD[J]. arXiv preprint arXiv:1908.07643, 2019.
[2] Thakkar O, Andrew G, McMahan H B. Differentially private learning with adaptive clipping[J]. arXiv preprint arXiv:1905.03871, 2019.
