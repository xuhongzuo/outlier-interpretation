# Outlier Interpretation

This repository contains the source code for the paper **Beyond Outlier Detection: Interpreting Outliers by  Attention-Guided Triplet Deviation Network** published in the Web Conference (WWW'21).   

Note that this task is also referred to as outlier explanation, outlier aspect mining/discovering, outlier property detection, and outlier description.



### Seven Outlier Interpretation Methods

**This repository contains seven outlier interpretation methods: ATON [1], COIN[2], SiNNE[3], SHAP[4], LIME[5], Integrated Gradients [6], and Anchor [7].**

[1] Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network. In WWW. 2021.

[2] Contextual outlier interpretation. In IJCAI. 2018.

[3] A new effective and efficient measure for outlying aspect mining. arXiv preprint arXiv:2004.13550. 2020.

[4] A unified approach to interpreting model predictions. In NeuraIPS. 2017

[5] "Why should I trust you?" Explaining the predictions of any classifier. In SIGKDD. 2016.

[6] Axiomatic attribution for deep networks. In ICML. 2017.

[7] Anchors: High Precision Model-Agnostic Explanations. In AAAI. 2018.



### Structure
`data_od_evaluation`: Ground-truth outlier interpretation annotations of real-world datasets  
`data`: real-world datasets in csv format, the last column is label indicating each line is an outlier or a inlier  
`model_xx`: folders of ATON and its contenders, the competitors are introduced in Section 5.1.2  
`config.py`: configuration and default hyper-parameters  
`main.py` main script to run the experiments



### How to use?
##### 1. For ATON and competitor COIN, SHAP, and LIME, and IntGrad
1. modify variant `algorithm_name` in `main.py` (support algorithm: `aton`, `coin`, `shap`, `lime`  in lowercase)
2. use `python main.py --path data/ --runs 10 `
3. the results can be found in `record/[algorithm_name]/` folder  

##### 2. For ATON' and competitor COIN' 
1. modify variant `algorithm_name` in `main.py` to `aton` or `coin`  
2. use `python main.py --path data/ --w2s_ratio auto --runs 10` to run ATON'  
   use `python main.py --path data/ --w2s_ratio pn --runs 10` to run COIN'  

##### 3. For competitor SiNNE and Anchor
1. modify variant `algorithm_name` in `main2.py` to `sinne` or `anchor`  
please run `python main2.py --path data/ --runs 10` 



### args of main.py
- `--path [str]`        - the path of data folder or an individual data file (in csv format)  

- `--gpu  [True/False]` - use GPU or not

- `--runs [int]`         - how many times to run a method on each dataset (we run 10 times and report average performance in our submission)

- `--w2s_ratio [auto/real_len/pn]`  - how to transfer feature weight to feature subspace 'real-len', 'auto', or 'pn' 
denote the same length with the ground-truth, auto generating subspace by the proposed threshold or positive-negative.
(in our paper, we use 'pn' in COIN', use 'auto' in ATON'. As for methods which output, we directly use 'real-len'.)

- `--eval [True/False]` - evaluate or not, use False for scalability test  
  ... (other hypter-parameters of different methods. You may want to use -h to check the corresponding hypter-parameters after modifing the `algorithm_name`)  

  

### Requirements
main packages of this project  
```
torch==1.3.0
numpy==1.15.0
pandas==0.25.2
scikit-learn==0.23.1
pyod==0.8.2
tqdm==4.48.2
prettytable==0.7.2
shap==0.35.0
lime==0.2.0.1
alibi==0.5.5
```



### Ground-truth annotations

Please also find the Ground-truth outlier interpretation annotations in folder `data_od_evaluation`.   
*We expect these annotations can foster further possible reasearchs on this new practical probelm.*  

You may find that each dataset has three annotation files, please refer to the detailed annotation generation process in our submission. We detailedly introduced it in Section 5.1.4:  

**How to generate the ground-truth annotations:**
>  We employ three different kinds of representative outlier detection methods (i.e., ensemble-based method iForest, probability-based method COPOD, and distance-based method HBOS) to evaluate outlying degree of real outliers given every possible subspace. A good explanation for an outlier should be a high-contrast subspace that the outlier explicitly demonstrates its outlierness, and outlier detectors can easily and certainly predict it as an outlier in this subspace. Therefore, the ground-truth interpretation for each outlier is defined as the subspace that the outlier obtains the highest outlier score among all the possible subspaces.



### a typo in the paper



In the second page, "As shown in Figure 1 (a), the queried outlier is ..., and the interpretation is feature subspace **$\{f1, f2\}$**" should be **$\{f1, f3\}$**.

We appreciate @Zeyi Li (NJPU) for finding this typo.  



### References
- datasets are from ODDS, a outlier detection datasets library (http://odds.cs.stonybrook.edu/), and kaggle platform (https://www.kaggle.com/)
- the source code of competitor COIN is publicly available in github. 



### Citation

If you find this useful in your research, please consider citing:
```
@inproceedings{xu2021aton,
	title={Beyond Outlier Detection: Interpreting Outliers by  Attention-Guided Triplet Deviation Network},
	author={Xu, Hongzuo and Wang, Yijie and Jian, Songlei and Huang, Zhenyu and Wang, Yongjun and Liu, Ning and Li, Fei},
	booktitle={Proceedings of The Web Conference 2021 (WWW’21)},
	year={2021},
	publisher={ACM}
}
```
