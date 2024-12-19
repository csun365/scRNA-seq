import streamlit as st
from analysis_script import *

def on_button_click():
    st.write("\nInitializing Analyzer...")
    sc_analysis = scRNAseqAnalysis("LT-HSC", "data/filtered_feature_bc_matrix_LT_HSC.h5", 
                                    "ST-HSC", "data/filtered_feature_bc_matrix_ST_HSC.h5",
                                    min_genes=1000, min_cells=50)
    st.write("Creating UMAP...")
    umap_fig = sc_analysis.umap(visible=True)
    st.session_state.umap_fig = umap_fig
    st.pyplot(umap_fig)
    st.write("Conducting Differential Expression Analysis...")
    sc_analysis.diff_exp()
    return sc_analysis

st.title("Single Cell RNA-seq Analysis Pipeline")

if "sc_analysis" not in st.session_state:
    st.session_state.sc_analysis = None

if "umap_fig" not in st.session_state:
    if st.button("Analyze LT-HSC vs. ST-HSC Transcriptome"):
        st.session_state.sc_analysis = on_button_click()
        st.session_state.all_genes = list(st.session_state.sc_analysis.cellxgene.columns)

if "umap_fig" in st.session_state:
    st.pyplot(st.session_state.umap_fig)
    
if st.session_state.sc_analysis:
    gene_name = st.text_input("Search gene (press enter):")
    if gene_name and len(gene_name) > 2:
        gene_name = gene_name[0].upper() + gene_name[1:].lower()
        st.write(f"Searching for {gene_name}...")
        if gene_name in st.session_state.all_genes:
            st.write(f"Generating UMAP for {gene_name}...")
            fig1 = st.session_state.sc_analysis.lookup_gene_umap(gene_name)
            st.pyplot(fig1)
            plt.figure()
            st.write(f"Fetching differential expression results for {gene_name}...")
            fig2, df = st.session_state.sc_analysis.lookup_gene_diff_exp(gene_name)
            st.dataframe(df)
            st.pyplot(fig2)
        else:
            st.write(f"Gene {gene_name} not found in the dataset or is invalid.")