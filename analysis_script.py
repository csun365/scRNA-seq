import numpy as np
import pandas as pd 
from IPython.display import display
import matplotlib.pyplot as plt 
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

class scRNAseqAnalysis():
    def __init__(self, name1, path1, name2, path2, min_genes, min_cells):
        print("\n\n\nscRNAseq Analysis Pipeline\n\n\n", end="")
        self.name1 = name1
        self.name2 = name2
        self.load_data(name1, path1, name2, path2)
        self.cellxgene, self.conditions = self.run_qc(min_genes, min_cells)
        self.umap_result = None
        self.diff_exp_df = None
        
    def load_data(self, name1, path1, name2, path2):
        print("\tLoading Data...")
        self.df1 = sc.read_10x_h5(path1)
        self.df2 = sc.read_10x_h5(path2)
        self.df1.obs["condition"] = name1
        self.df2.obs["condition"] = name2
        
    def run_qc(self, min_genes, min_cells):
        print("\tRunning Quality Control...")

        # Filter by cells and genes
        sc.pp.filter_cells(self.df1, min_genes=min_genes)
        sc.pp.filter_cells(self.df2, min_genes=min_genes)
        sc.pp.filter_genes(self.df1, min_cells=min_cells)
        sc.pp.filter_genes(self.df2, min_cells=min_cells)

        # Join data frames
        self.df1.var_names_make_unique()
        self.df2.var_names_make_unique()   
        include_genes = self.df1.var_names.intersection(self.df2.var_names)
        self.df1 = self.df1[:,include_genes]
        self.df2 = self.df2[:,include_genes]
        cellxgene = self.df1.concatenate(self.df2)

        # Print data dimensionality
        print("\t\t", len(include_genes), "genes")
        print("\t\t", self.df1.obs.shape[0], self.name1)
        print("\t\t", self.df2.obs.shape[0], self.name2)
        
        # Compute normalized counts
        sc.pp.normalize_total(cellxgene)
        sc.pp.log1p(cellxgene)
        sc.pp.scale(cellxgene)
        
        # Assemble cell by gene matrix and condition labels
        features = pd.DataFrame(cellxgene.X, index=cellxgene.obs_names, columns=cellxgene.var_names)
        labels = np.array(list(self.df1.obs["condition"].values) + list(self.df2.obs["condition"].values))
        return features, labels

    def plot_2d(self, data, x_label, y_label):
        plt.scatter(
            data[self.conditions == self.name1,0], 
            data[self.conditions == self.name1,1], 
            c="b", label=self.name1, alpha=0.5, s=1
        ) 
        plt.scatter(
            data[self.conditions == self.name2,0], 
            data[self.conditions == self.name2,1], 
            c="g", label=self.name2, alpha=0.5, s=1
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        return plt

    def pca(self, n_components=2, visible=True):
        pca_obj = PCA(n_components=n_components)
        self.pca_result = pca_obj.fit_transform(self.cellxgene)
        if visible:
            print("Variance Maintained: ", pca_obj.explained_variance_ratio_)
            return self.plot_2d(self.pca_result, "PC1", "PC2")

    def tsne(self, n_components=2, visible=True):
        tsne_obj = TSNE(n_components=n_components, random_state=0)
        self.tsne_result = tsne_obj.fit_transform(self.cellxgene)
        if visible:
            return self.plot_2d(self.tsne_result, "t-SNE", "t-SNE")

    def umap(self, n_neighbors=15, visible=True):
        umap_obj = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.5, metric="euclidean", random_state=0)
        self.umap_result = umap_obj.fit_transform(self.cellxgene)
        if visible:
            return self.plot_2d(self.umap_result, "UMAP1", "UMAP2")

    def lookup_gene_umap(self, gene_name):
        if self.umap_result is None:
            self.umap(visible=False)
        colors = self.cellxgene[gene_name]
        plt.scatter(self.umap_result[:,0], self.umap_result[:,1], c=colors, s=1, cmap="Reds") 
        plt.colorbar()
        plt.title(gene_name)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        return plt
    
    def diff_exp(self, threshold=0.05):
        results = []
        for gene in self.cellxgene.columns:
            group1 = self.cellxgene.loc[self.conditions == self.name1, gene]
            group2 = self.cellxgene.loc[self.conditions == self.name2, gene]
            _, p_value = mannwhitneyu(group1, group2)
            log_fc = np.log2(np.mean(group1) + 1) - np.log2(np.mean(group2) + 1)
            results.append({"gene": gene, "log2FC": log_fc, "p_value": p_value})
        self.diff_exp_df = pd.DataFrame(results)
        
        # Benjamini-Hochberg
        self.diff_exp_df["adj_p_value"] = self.diff_exp_df["p_value"].rank(method="first") / (len(results) * threshold)
        self.sig_genes = self.diff_exp_df[self.diff_exp_df["adj_p_value"] < threshold]
    
    def lookup_gene_diff_exp(self, gene_name):
        if self.diff_exp_df is None:
            self.diff_exp()
        
        # Display differential expression statistics
        display(self.diff_exp_df[self.diff_exp_df["gene"] == gene_name])

        # Plot histogram of expression across all cells
        group1 = self.cellxgene.loc[self.conditions == self.name1, gene_name]
        group2 = self.cellxgene.loc[self.conditions == self.name2, gene_name]
        plt.hist(group1, weights=np.ones_like(group1) / group1.shape[0], alpha=0.5, color="r", label=self.name1)
        plt.hist(group2, weights=np.ones_like(group2) / group2.shape[0], alpha=0.5, color="b", label=self.name2)
        plt.title(gene_name)
        plt.xlabel("Normalized Expression")
        plt.ylabel("Relative Frequency")
        plt.legend()
        return plt, self.diff_exp_df[self.diff_exp_df["gene"] == gene_name]

    def volcano(self, fc_threshold=2, p_threshold=0.05):
        if self.diff_exp_df is None:
            self.diff_exp()
        down_reg = self.diff_exp_df[
            (self.diff_exp_df["log2FC"] < -fc_threshold) & 
            (self.diff_exp_df["adj_p_value"] < p_threshold)
        ]
        up_reg = self.diff_exp_df[
            (self.diff_exp_df["log2FC"] > fc_threshold) & 
            (self.diff_exp_df["adj_p_value"] < p_threshold)
        ]
        plt.scatter(self.diff_exp_df["log2FC"], -np.log10(self.diff_exp_df["adj_p_value"]), s=1, c="k")
        plt.scatter(down_reg["log2FC"], -np.log10(down_reg["adj_p_value"]), s=1, label="Down-regulated", c="b")
        plt.scatter(up_reg["log2FC"], -np.log10(up_reg["adj_p_value"]), s=1, label="Up-regulated", c="r")

        genes = list(up_reg.sort_values("adj_p_value")["gene"].iloc[:5])
        for i, row in up_reg.iterrows():
            if row["gene"] in genes:
                plt.text(row["log2FC"], -np.log10(row["adj_p_value"]), s=row["gene"], fontsize=6)
                
        plt.xlabel("log$_2$FC")
        plt.ylabel("-log$_{10}$p-value")
        plt.axvline(-fc_threshold, c="grey", linestyle="--")
        plt.axvline(fc_threshold, c="grey", linestyle="--")
        plt.axhline(-np.log10(p_threshold), c="grey", linestyle="--")
        plt.legend(fontsize=8)
        plt.show()