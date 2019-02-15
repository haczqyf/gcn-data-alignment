import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alignment.classes import *
from alignment.utils import *

from alignment.measures.randomizations import *
from alignment.measures.subspaces import *
from alignment.exceptions import ScanningCountNotPossible


def optimize_dim_subspaces(dataset, num_rdm, num_k, num_scanning=1, norm_type="Frobenius-Norm", log=True, heatmap=False):
    """
    Find dimensions for features and graph by optimization.
    
    Parameters:
        dataset(string): dataset name
        num_rdm(int): number of instances for randomization
        num_k(int): number of dimensions evenly split
        num_scanning(int): number of scanning rounds
        log(boolean): decide if the log is printed
        heatmap(boolean): decide if the heatmap is plotted

    Returns:
        Correspoding optimized dimensions for features, graph and ground truth
    Return type:
        Dictionary
    """
    if num_scanning > 2:
        raise ScanningCountNotPossible(
            "The num_scanning can only be 1 or 2.")

    node_l, G, X, A, Y = load_data(dataset=dataset)
    igds = Ingredients(nodelist=node_l, G=G, X=X, A=A, Y=Y)

    k_Y = Y.shape[1]
    opt_results = {'k_X': k_Y, 'k_A': k_Y, 'k_Y': k_Y}

    for idx in range(num_scanning):
        print("Scanning round {}".format(idx+1))
        if idx == 0:
            k_X_l=[int(x) for x in np.linspace(Y.shape[1], X.shape[1], num=num_k)]
            k_A_l=[int(x) for x in np.linspace(Y.shape[1], A.shape[1], num=num_k)]
        else:
            k_X_opt_index = k_X_l.index(opt_results['k_X'])
            if k_X_opt_index == 0:
                k_X_l_min = opt_results['k_X']
                k_X_l_max = k_X_l[1]
            elif k_X_opt_index == num_k - 1:
                k_X_l_min = k_X_l[-2]
                k_X_l_max = opt_results['k_X']
            else:
                k_X_l_min = k_X_l[k_X_opt_index-1]
                k_X_l_max = k_X_l[k_X_opt_index+1]

            k_A_opt_index = k_A_l.index(opt_results['k_A'])
            if k_A_opt_index == 0:
                k_A_l_min = opt_results['k_Y']
                k_A_l_max = k_A_l[1]
            elif k_Y_opt_index == num_k - 1:
                k_A_l_min = k_A_l[-2]
                k_A_l_max = opt_results['k_Y']
            else:
                k_A_l_min = k_A_l[k_A_opt_index-1]
                k_A_l_max = k_A_l[k_A_opt_index+1]

            k_X_l=[int(x) for x in np.linspace(k_X_l_min, k_X_l_max, num=num_k)]
            k_A_l=[int(x) for x in np.linspace(k_A_l_min, k_A_l_max, num=num_k)]

        print("k_X_l: " + str(k_X_l))
        print("k_A_l: " + str(k_A_l))

        df = pd.DataFrame(columns=['k_X','k_A','k_Y','d_zero_rdm','d_full_rdm','d_diff_zero_full_rdm'])
        i = 0

        for k_X in k_X_l:
            for k_A in k_A_l:
                d_zero_rdm = distance(
                                X=igds.get_X_gcn(p=0), 
                                A=igds.get_A_gcn(p=0), 
                                Y=igds.get_Y_gcn(), 
                                k_X=k_X, 
                                k_A=k_A, 
                                k_Y=k_Y,
                                norm_type=norm_type
                                )
                d_full_rdm = 0
                for j in range(num_rdm):
                    d_full_rdm_temp = distance(
                                        X=igds.get_X_gcn(p=100), 
                                        A=igds.get_A_gcn(p=100), 
                                        Y=igds.get_Y_gcn(), 
                                        k_X=k_X, 
                                        k_A=k_A, 
                                        k_Y=k_Y,
                                        norm_type=norm_type
                                        )
                    if log == True:
                        print("k_X={},k_A={},d_zero_rdm={},random_id={},d_full_rdm_temp={}".format(
                            k_X,
                            k_A,
                            d_zero_rdm,
                            j,
                            d_full_rdm_temp))
                    d_full_rdm = d_full_rdm + d_full_rdm_temp
                
                d_full_rdm = d_full_rdm / num_rdm
                df.loc[i] = [k_X,k_A,k_Y,d_zero_rdm,d_full_rdm,d_full_rdm-d_zero_rdm]
                i = i + 1

        # Plotting heatmap
        if heatmap == True:
            df.k_X = df.k_X.astype(int)
            df.k_A = df.k_A.astype(int)
            df.k_Y = df.k_Y.astype(int)

            piv = pd.pivot_table(df, values="d_diff_zero_full_rdm",index=["k_A"], columns=["k_X"], fill_value=0)

            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(piv, cmap="Greens")
            fig.colorbar(im, ax=ax)

            ax.set_xticks(range(len(piv.columns)))
            ax.set_yticks(range(len(piv.index)))
            ax.set_xticklabels(piv.columns, rotation=90)
            ax.set_yticklabels(piv.index)
            ax.set_xlabel(r"$k_{X}$",fontsize=16)
            ax.set_ylabel(r"$k_{A}$",fontsize=16)
            plt.savefig('{}_scanning_round{}.eps'.format(dataset,idx+1))

            plt.tight_layout()
            plt.show()

        opt_results = dict(df.iloc[df['d_diff_zero_full_rdm'].idxmax()].astype(int)[["k_X","k_A","k_Y"]])
        print("Optimization results in scanning round {}: {}".format(idx+1,opt_results))
        print("\n")

    return opt_results

if __name__ == "__main__":
    print(optimize_dim_subspaces(
        dataset="constructive_example",
        num_rdm=1,
        num_k=5,
        num_scanning=2,
        norm_type="Frobenius-Norm",
        log=True,
        heatmap=True
        ))
    