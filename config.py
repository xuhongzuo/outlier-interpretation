# ------------------- path of datasets and annotations ----------------- #
root = ''
eva_root = ''


def get_parser(algorithm_name, parser):
    if algorithm_name == "aton":
        parser.add_argument('--nbrs_num', type=int, default=30, help='')
        parser.add_argument('--rand_num', type=int, default=30, help='')
        parser.add_argument('--alpha1', type=float, default=0.8, help='triplet loss factor in loss function')
        parser.add_argument('--alpha2', type=float, default=0.2, help='dis loss factor in loss function')
        parser.add_argument('--n_epoch', type=int, default=10, help='')
        parser.add_argument('--batch_size', type=int, default=512, help='')
        parser.add_argument('--lr', type=float, default=0.1, help='')
        parser.add_argument('--n_linear', type=int, default=64, help='')
        parser.add_argument('--margin', type=float, default=5., help='')
    elif algorithm_name == "shap":
        parser.add_argument('--kernel', type=str, default='rbf', help='')
        parser.add_argument("--n_sample", type=int, default=100, help='')
        parser.add_argument("--threshold", type=int, default=-1, help='')
    elif algorithm_name == "lime":
        parser.add_argument('--discretize_continuous', type=bool, default=True, help='')
        parser.add_argument("--discretizer", type=str, default="quartile", help='')
    elif algorithm_name == "intgrad":
        parser.add_argument('--n_steps', type=int, default=40, help='')
        parser.add_argument('--method', type=str, default="gausslegendre", help='')
    elif algorithm_name == "coin":
        parser.add_argument('--AUG', type=float, default=10, help='an additional attribute value as augmentation')
        parser.add_argument('--ratio_nbr', type=float, default=0.08,
                            help='controls number of neighbors to use in kneighbors queries')
        parser.add_argument('--MIN_CLUSTER_SIZE', type=int, default=5,
                            help='minimum number of samples required in a cluster')
        parser.add_argument('--MAX_NUM_CLUSTER', type=int, default=4,
                            help='maximum number of clusters for each context')
        parser.add_argument('--VAL_TIMES', type=int, default=10,
                            help='number of iterations for computing prediction strength')
        parser.add_argument('--C_SVM', type=float, default=1., help='penalty parameter for svm')
        parser.add_argument('--DEFK', type=int, default=0,
                            help='pre-determined number of clusters in each context (use prediction strength if 0)')
        parser.add_argument('--THRE_PS', type=float, default=0.85,
                            help='threshold for deciding the best cluster value in prediction strength')
    elif algorithm_name == "aton_ablation" or algorithm_name == "aton_ablation2" or algorithm_name == "aton_ablation3":
        parser.add_argument('--nbrs_num', type=int, default=30, help='')
        parser.add_argument('--rand_num', type=int, default=30, help='')
        parser.add_argument('--n_epoch', type=int, default=10, help='')
        parser.add_argument('--batch_size', type=int, default=64, help='')
        parser.add_argument('--lr', type=float, default=0.1, help='')
        parser.add_argument('--n_linear', type=int, default=64, help='')
        parser.add_argument('--margin', type=float, default=5., help='')
    elif algorithm_name == "sinne":
        parser.add_argument('--max_level', default='full', help='')
        parser.add_argument("--width", type=int, default=10, help='')
        parser.add_argument("--ensemble_num", type=int, default=100, help='')
        parser.add_argument("--sample_num", type=int, default=8, help='')
        parser.add_argument("--pretrain", type=bool, default=False, help='')
        parser.add_argument("--verbose", type=bool, default=False, help='')
    else:
        raise NotImplementedError("not supported algorithm")
    return parser

