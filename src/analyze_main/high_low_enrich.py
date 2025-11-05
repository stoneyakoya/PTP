import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

# from umap import UMAP
from sklearn.metrics import silhouette_samples, mean_squared_error, r2_score
from gprofiler import GProfiler
from matplotlib.patches import Patch


# ==============================
# データ読み込み・前処理
# ==============================
def load_and_merge_data(
    cluster_info_path: str,
    result_base_dir: str,
    n_folds: int = 10
) -> pd.DataFrame:
    """
    ・cluster_info_path: 'data/tmp/input_target_df_add_cluster_info.csv' のように、ペプチド情報とクラスタ番号が入ったCSV。
    ・result_base_dir: 'data/results/03_AASeq_Transformer_based_sampling' のように、fold_i/non_recursive.csv が置かれているディレクトリのパス。
      各 fold_i の下に non_recursive.csv がある前提。

    戻り値:
      df_all: 以下のカラムを持つ DataFrame
        - Cleaned_Peptidoform (ペプチドID)
        - UniProt IDs
        - Peptido_Embedding_Path
        - Protein Sequence
        - Cluster
        - Peptide_ID (Cleaned_Peptidoform と同じ)
        - R2_Score (各ペプチドでのR²スコア)
    """
    # 1-1) クラスタ情報を読み込み
    input_df = pd.read_csv(cluster_info_path)
    # 必要なカラムだけ抜き出し, unique
    keep_cols = ['Cleaned_Peptidoform', 'UniProt IDs', 'Peptido_Embedding_Path', 'Protein Sequence', 'Cluster']
    target_df = input_df[keep_cols].drop_duplicates().copy()
    target_df.rename(columns={'Cleaned_Peptidoform': 'Peptide_ID'}, inplace=True)

    # 1-2) 各 fold_i の結果をまとめて読み込み
    all_fold_results = []
    for i in range(n_folds):
        path_i = os.path.join(result_base_dir, f'fold_{i}', 'non_recursive.csv')
        if not os.path.exists(path_i):
            raise FileNotFoundError(f"Fold{i} の結果ファイルが見つかりません: {path_i}")
        df_i = pd.read_csv(path_i)
        # Predictions_Turnover が文字列 "[0.123]" のような形式になっている想定 → 数値に変換
        df_i['Predictions_Turnover'] = df_i['Predictions_Turnover'].apply(lambda x: float(x.strip('[]')))
        # ペプチドID列は 'Peptide_ID' というカラム名で揃えておく（ファイル内で違う場合は修正）
        if 'Peptide_ID' not in df_i.columns and 'Cleaned_Peptidoform' in df_i.columns:
            df_i.rename(columns={'Cleaned_Peptidoform': 'Peptide_ID'}, inplace=True)
        all_fold_results.append(df_i)

    # 結合
    df_concat = pd.concat(all_fold_results, axis=0, ignore_index=True)
    
    # 1-3) ペプチドごとにR²スコアを計算
    per_peptide_r2 = []
    for peptide_id, group in df_concat.groupby('Peptide_ID'):
        # 真値と予測値を取得
        y_true = group['True_Values_Turnover'].values
        y_pred = group['Predictions_Turnover'].values
        
        # R²スコアを計算（サンプル数が2以上の場合のみ）
        if len(y_true) >= 2:
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = np.nan
        else:
            r2 = np.nan
        
        per_peptide_r2.append({
            'Peptide_ID': peptide_id,
            'R2_Score': r2,
            'n_samples': len(y_true)
        })
    
    df_r2 = pd.DataFrame(per_peptide_r2)

    # 1-4) クラスタ情報＋埋め込みパス＋R²スコアをマージ
    df_all = pd.merge(target_df, df_r2, on='Peptide_ID', how='inner')
    # 埋め込みファイルパスがないものは除外
    df_all = df_all[df_all['Peptido_Embedding_Path'].notna()].reset_index(drop=True)
    # Cleaned_Peptidoform カラムを復元（元の形式に合わせるため）
    df_all['Cleaned_Peptidoform'] = df_all['Peptide_ID']
    
    return df_all


# ==============================
# クラスターごとの高精度ペプチドを抽出
# ==============================
def extract_high_performance_peptides(df: pd.DataFrame, cluster_id: int, top_percentile: float = 0.9) -> pd.DataFrame:
    """
    特定のクラスタ内でR²スコアの上位パーセンタイルのペプチドを抽出
    
    Parameters:
    -----------
    df : pd.DataFrame
        全データ
    cluster_id : int
        対象のクラスタID
    top_percentile : float
        上位パーセンタイル（デフォルト: 0.9 = 上位10%）
    
    Returns:
    --------
    pd.DataFrame
        高精度ペプチドのデータフレーム
    """
    # 特定のクラスタのみ抽出
    df_cluster = df[df['Cluster'] == cluster_id].copy()
    
    # R²スコアでパーセンタイルを計算
    threshold = df_cluster['R2_Score'].quantile(top_percentile)
    
    # 高精度ペプチドを抽出
    df_high = df_cluster[df_cluster['R2_Score'] >= threshold].copy()
    
    print(f"クラスタ {cluster_id} の高精度ペプチド:")
    print(f"  全ペプチド数: {len(df_cluster)}")
    print(f"  高精度ペプチド数 (R² >= {threshold:.3f}): {len(df_high)}")
    print(f"  R²スコアの平均: {df_high['R2_Score'].mean():.3f}")
    print()
    
    return df_high


# ==============================
# エンリッチメント解析
# ==============================
def run_enrichment_analysis(
    df_peptides: pd.DataFrame,
    organism: str = 'hsapiens',
    sources: list = ['GO:BP', 'GO:MF', 'GO:CC', 'KEGG'],
    top_n: int = 20
) -> pd.DataFrame:
    """
    ペプチドリストからUniProt IDを抽出してエンリッチメント解析を実行
    
    Parameters:
    -----------
    df_peptides : pd.DataFrame
        ペプチドのデータフレーム（UniProt IDsカラムを含む）
    organism : str
        生物種
    sources : list
        使用するデータベース
    top_n : int
        上位何件を返すか
    
    Returns:
    --------
    pd.DataFrame
        エンリッチメント解析の結果（上位top_n件）
    """
    # UniProt IDを抽出（セミコロン区切りの最初のIDのみ使用）
    uniprot_ids = list({uid.split(';')[0] for uid in df_peptides['UniProt IDs'].dropna().values})
    
    print(f"  UniProt ID数: {len(uniprot_ids)}")
    
    # g:Profilerでエンリッチメント解析
    gp = GProfiler(return_dataframe=True)
    try:
        results = gp.profile(organism=organism, query=uniprot_ids, sources=sources)
        if top_n is not None:
            results = results.sort_values('p_value', ascending=True).head(top_n)
        else:
            results = results.sort_values('p_value', ascending=True)
        results['minus_log10_p'] = -np.log10(results['p_value'])
        return results
    except Exception as e:
        print(f"  エンリッチメント解析エラー: {e}")
        return pd.DataFrame()


# ==============================
# 複数クラスター比較バブルプロット
# ==============================
def create_comparison_bubble_plot(
    enrichment_results: dict,
    top_n_per_cluster: int = 10,
    x_label_prefix: str = 'Cluster',
    title: str = 'Enrichment Analysis',
    save_path: str = 'data/data_analyze/enrichment/cluster_comparison_enrichment_bubble.pdf'
) -> None:
    """
    複数クラスターのエンリッチメント結果を比較するバブルプロットを作成
    
    Parameters:
    -----------
    enrichment_results : dict
        {cluster_id: enrichment_dataframe} の形式
    out_dir : str
        出力ディレクトリ
    top_n_per_cluster : int
        各クラスターから表示する上位term数
    """

    # 1. 手動カテゴリ辞書
    term_category_dict = {
        # --- Cell Cycle / Cytoskeleton ---
        'cell cycle process': 'Cell Cycle',
        'mitotic cell cycle': 'Cell Cycle',
        'mitotic cell cycle process': 'Cell Cycle',
        'spindle': 'Cell Cycle',
        'microtubule cytoskeleton': 'Cell Cycle',
        'chromosome': 'Cell Cycle',
        'nucleoplasm': 'Cell Cycle',

        # --- Organelle / Lumen / Vesicle ---
        'membraneless organelle': 'Organelle/Lumen',
        'intracellular membraneless organelle': 'Organelle/Lumen',
        'organelle': 'Organelle/Lumen',
        'intracellular organelle lumen': 'Organelle/Lumen',
        'membrane-enclosed lumen': 'Organelle/Lumen',
        'extracellular organelle': 'Organelle/Lumen',
        'extracellular vesicle': 'Organelle/Lumen',
        'extracellular exosome': 'Organelle/Lumen',
        'organelle lumen': 'Organelle/Lumen',
        'intracellular organelle lumen': 'Organelle/Lumen',
        'membrane-enclosed lumen': 'Organelle/Lumen',
        'extracellular membrane-bounded organelle': 'Organelle/Lumen',
        'nuclear lumen': 'Organelle/Lumen',

        # --- Nucleotide / Metabolic ---
        'nucleotide binding': 'Nucleotide/Metabolic',
        'purine nucleotide binding': 'Nucleotide/Metabolic',
        'nucleoside phosphate binding': 'Nucleotide/Metabolic',
        'adenyl nucleotide binding': 'Nucleotide/Metabolic',
        'heterocyclic compound binding': 'Nucleotide/Metabolic',
        'catalytic activity': 'Nucleotide/Metabolic',

        # --- Generic Binding ---
        'protein binding': 'Binding',
        'RNA binding': 'Binding',

        # --- Localization ---
        'cytosol': 'Cytosol',
        'cytoplasm': 'Cytosol',
    }

    # 2. カテゴリ→色
    category_colors = {
    'Cell Cycle':       '#007acc',  # 鮮やかな青（中〜濃めのスカイブルー）
    'Organelle/Lumen':  '#ffcc00',  # 明るくはっきりした黄色（ゴールデンイエロー）
    'Nucleotide/Metabolic': '#33aa33',  # 鮮やかな緑（フレッシュなリーフグリーン）
    'Other':            '#bbbbbb',  # 少し明るめのグレー
}


    
    
    # 1) 各クラスターから上位N個のtermを取得
    cluster_ids = sorted(enrichment_results.keys())
    all_terms = set()
    cluster_top_terms = {}
    
    for cluster_id, df_enrich in enrichment_results.items():
        if df_enrich.empty:
            continue
        # 各クラスターの上位termを取得
        top_terms = df_enrich.nlargest(top_n_per_cluster, 'minus_log10_p')
        cluster_top_terms[cluster_id] = top_terms
        all_terms.update(top_terms['name'].tolist())
    
    # 全てのユニークなtermのリストを作成
    all_terms_list = sorted(list(all_terms))
    
    # 2) プロット用データの準備
    plot_data = []
    for cluster_id in cluster_ids:
        if cluster_id not in cluster_top_terms:
            continue
        df_cluster = cluster_top_terms[cluster_id]
        for _, row in df_cluster.iterrows():
            plot_data.append({
                'cluster_id': cluster_id,
                'name': row['name'],
                'minus_log10_p': row['minus_log10_p'],
                'intersection_size': row['intersection_size']
            })
    
    df_plot = pd.DataFrame(plot_data)

    df_plot['category'] = df_plot['name'].map(term_category_dict).fillna('Other')
    df_plot['category_color'] = df_plot['category'].map(category_colors)
    
    # 3) termごとにy座標を割り当て（有意性の高い順）
    term_scores = df_plot.groupby('name')['minus_log10_p'].max().sort_values(ascending=False)
    term_to_y = {term: i for i, term in enumerate(term_scores.index)}
    df_plot['y_pos'] = df_plot['name'].map(term_to_y)
    
    # x座標はクラスターごとに割り当て（間隔を広げて重なりを防ぐ）
    cluster_spacing = 0.8  # クラスター間の間隔
    cluster_to_x = {cluster_id: i * cluster_spacing for i, cluster_id in enumerate(cluster_ids)}
    df_plot['x_pos'] = df_plot['cluster_id'].map(cluster_to_x)
    
    # 4) プロット作成（図のサイズを調整）
    n_terms = len(term_to_y)
    fig_height = max(13, n_terms * 0.4)  # termが多い場合は高さを増やす
    fig, ax = plt.subplots(figsize=(50, fig_height))
    # 右側に 28 % の余白を確保（0.72 までがメイン軸領域）
    fig.subplots_adjust(left=0.40,   # ← y 軸ラベル用に左 40 %
                    right=0.5,  # ← 右 32 % を丸ごと空白帯に
                    top=0.95,
                    bottom=0.05)


    # スタイル設定
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    # カラーマップとノーマライズ
    vmin = 0
    vmax = df_plot['minus_log10_p'].max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    
    # バブルサイズのスケール設定（重なりを考慮して調整）
    max_size = df_plot['intersection_size'].max()
    min_size = df_plot['intersection_size'].min()
    # サイズの範囲を150-2000に設定（適度な大きさ）
    size_scale = lambda x: 150 + (x - min_size) / (max_size - min_size) * 1850 if max_size > min_size else 1000
    
    # 各クラスターごとにプロット
    for cluster_id in cluster_ids:
        df_cluster = df_plot[df_plot['cluster_id'] == cluster_id]
        if df_cluster.empty:
            continue
            
        scatter = ax.scatter(
            x=df_cluster['x_pos'],
            y=df_cluster['y_pos'],
            s=df_cluster['intersection_size'].apply(size_scale),
            c=df_cluster['minus_log10_p'],
            cmap=cmap,
            norm=norm,
            alpha=0.85,
            edgecolor='black',
            linewidth=1.0,
            label=f'Cluster {cluster_id}'
        )
    
    # グリッドとスタイル設定
    ax.grid(True, linestyle=':', alpha=0.4, axis='y', color='gray', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # x軸の設定（クラスター間の距離を調整）
    ax.set_xticks([cluster_to_x[cid] for cid in cluster_ids])
    if x_label_prefix:
        xtlabs = [f'{x_label_prefix} {cid}' for cid in cluster_ids]
    else:
        xtlabs = [str(cid) for cid in cluster_ids]   
    ax.set_xticklabels([f'Cluster {cid}' for cid in cluster_ids], fontsize=28, weight='bold')
    ax.set_xticklabels([f'{cid}' for cid in cluster_ids], fontsize=28, weight='bold')

    ax.set_xlim(-0.4, max(cluster_to_x.values()) + 0.4)
    
    # y軸の設定（GO term名の処理）
    y_labels = []
    y_colors  = []     
    for term in term_scores.index:
        # 短めに調整
        if len(term) > 45:
            term = term[:42] + '...'
        y_labels.append(term)
        cat  = term_category_dict.get(term, 'Other')
        y_colors.append(category_colors.get(cat, category_colors['Other']))
    
    ax.set_yticks(range(len(term_scores)))
    ax.set_yticklabels(y_labels, fontsize=24, ha='right')
    ax.set_ylim(-0.5, len(term_scores) - 0.5)
    
    # y軸の余白を増やす
    ax.tick_params(axis='y', pad=5)
    
    # ラベル
    ax.set_xlabel('', fontsize=26, weight='bold')
    ax.set_ylabel('GO Terms / Pathways', fontsize=26, weight='bold', labelpad=15)
    ax.set_title(title, 
                fontsize=30, weight='bold', pad=30)
    # ↓ ここを追加
    for label, color in zip(ax.get_yticklabels(), y_colors):
        label.set_fontsize(28)
        label.set_weight('bold')
        label.set_color(color)
        # シャドウ効果
        label.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground='white'),
            path_effects.Normal()
        ])

    # カテゴリごとの背景色を定義
    category_bg_colors = {
        'Cell Cycle': '#E6F3FF',  # 薄い青
        'Organelle/Lumen': '#FFF9E6',  # 薄い黄色
        'Nucleotide/Metabolic': '#E6FFE6',  # 薄い緑
        'Other': '#F5F5F5',  # 薄いグレー
    }

    # カラーバーの調整
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        orientation='vertical',
        pad=0.12,
        fraction=0.04,
        aspect=20,
        shrink=0.7
    )
    cbar.set_label('-log₁₀(p-value)', rotation=270, labelpad=25, fontsize=22, weight='bold')
    cbar.ax.tick_params(labelsize=18)

    # 凡例のスタイル調整（枠なしに変更）
    cat_handles = [Patch(color=c, label=lab) for lab, c in category_colors.items()]
    cats_legend = ax.legend(
        handles=cat_handles,
        title='Functional Category\n(label colour)',
        loc='center left',
        borderpad=1.5,
        fontsize=18,
        title_fontsize=20,
        edgecolor='black',
        bbox_to_anchor=(-2.0, 0.9),
        frameon=False,  # 枠を消す
    )
    
    # サイズの凡例（右側の適切な位置に配置）
    # 実際のデータに基づいたサイズを表示
    size_values = [df_plot['intersection_size'].min(), 
                   df_plot['intersection_size'].median(), 
                   df_plot['intersection_size'].max()]
    size_handles = []
    size_labels = []
    
    for val in size_values:
        size = size_scale(val)
        handle = plt.scatter([], [], s=size, c='lightgray', alpha=0.8, edgecolor='black', linewidth=1.5)
        size_handles.append(handle)
        size_labels.append(f'{int(val)}')
    
    # サイズ凡例を右側に配置
    size_legend = ax.legend(
        size_handles, size_labels,
        title='Gene Count',
        loc='center left',
        bbox_to_anchor=(1.15, 0.9),
        frameon=False,
        fancybox=False,
        shadow=False,
        title_fontsize=20,
        columnspacing=1.0,           # 列間
        fontsize=18,
        borderpad=1.5,
        handletextpad=1,
        markerscale=1.0,
        edgecolor='black',
        labelspacing=2.0,
    )
    size_legend.get_title().set_weight('bold')
    cats_legend.get_title().set_weight('bold')
    ax.add_artist(size_legend)
    ax.add_artist(cats_legend)
    
    # 左側の余白を調整してGO term名が切れないようにする
    # plt.subplots_adjust(left=0.4, right=0.85, top=0.95, bottom=0.05)
    
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', format='pdf')
    plt.show()
    print(f"\n保存完了: {save_path}")

def prepare_hi_lo_enrichment(df_all, cluster_id: int,
                             hi_pct=0.9, lo_pct=0.1,
                             top_n=15) -> dict:
    cl_df = df_all.query("Cluster == @cluster_id").copy()
    hi_thr = cl_df["R2_Score"].quantile(hi_pct)
    lo_thr = cl_df["R2_Score"].quantile(lo_pct)

    hi_df = cl_df.query("R2_Score >= @hi_thr")
    lo_df = cl_df.query("R2_Score <= @lo_thr")

    return {
        'High': run_enrichment_analysis(hi_df, top_n=top_n),
        'Low' : run_enrichment_analysis(lo_df, top_n=top_n)
    }


def save_enrichment_result_for_supplementary(enrichment_results: dict, output_dir: str = "supplementary"):
    """
    各クラスタのエンリッチメント結果を論文用の整形CSVとして保存する関数。

    Parameters:
    -----------
    enrichment_results : dict
        {cluster_name: enrichment_result_df} の辞書
    output_dir : str
        保存先ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster_name, df in enrichment_results.items():
        if df.empty:
            print(f"[!] クラスタ {cluster_name} は空のためスキップ")
            continue

        # 抽出するカラム（存在するものだけ使う）
        expected_cols = {
            "term_id": "Term ID",
            "name": "GO Term / Pathway Name",
            "source": "Source",
            "p_value": "P-value",
            "intersection_size": "Gene Count",
            "minus_log10_p": "-log10(p-value)"
        }
        actual_cols = [col for col in expected_cols if col in df.columns]
        renamed_cols = {col: expected_cols[col] for col in actual_cols}

        # 並べ替え＋rename
        df_out = df[actual_cols].copy()
        df_out = df_out.rename(columns=renamed_cols)

        # ソート（p値昇順）
        df_out = df_out.sort_values(by="P-value")

        # 保存
        output_path = os.path.join(output_dir, f"enrichment_VeryRapid_high&low_{cluster_name.replace(' ', '_').lower()}.csv")
        df_out.to_csv(output_path, index=False)
        print(f"[✓] Saved: {output_path}")

# ==============================
# メイン処理
# ==============================
def main():
    # データを読み込み
    df_all = load_and_merge_data(
        cluster_info_path='data/tmp/input_target_df_add_cluster_info.csv',
        result_base_dir='data/results/03_AASeq_Transformer_based_sampling',
        n_folds=10
    )
    
    print("データ読み込み完了")
    print(f"総ペプチド数: {len(df_all)}")
    print(f"カラム: {list(df_all.columns)}")
    print(f"クラスター分布:\n{df_all['Cluster'].value_counts().sort_index()}")
    print("\n" + "="*50 + "\n")
    
    # クラスター4と1の高精度ペプチドを抽出
    clusters_to_analyze = [4, 1]
    enrichment_results = {}
    
    for cluster_id in clusters_to_analyze:
        print(f"--- クラスター {cluster_id} の解析 ---")
        
        # 高精度ペプチドを抽出（上位10%）
        df_high = extract_high_performance_peptides(
            df_all, 
            cluster_id=cluster_id, 
            top_percentile=0.9
        )
        
        # エンリッチメント解析
        print(f"エンリッチメント解析を実行中...")
        enrichment_df = run_enrichment_analysis(
            df_high,
            organism='hsapiens',
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG'],
            top_n=None

        )
        cluster_name_dict = {4:'Very Rapid', 1:'Very Slow'}
        if not enrichment_df.empty:
            enrichment_results[cluster_name_dict[cluster_id]] = enrichment_df
            print(f"上位20件のGO term:")
            print(enrichment_df[['name', 'p_value', 'intersection_size']])
        
        print("\n" + "="*50 + "\n")
    
    # 比較バブルプロットを作成
    if enrichment_results:
        print("比較バブルプロットを作成中...")
        create_comparison_bubble_plot(
            enrichment_results,
            top_n_per_cluster=13,
            title='Enrichment Analysis: Very Rapid Cluster vs Very Slow Cluster',
            save_path='data/data_analyze/enrichment/cluster_comparison_enrichment_bubble.pdf'
        )

        # Supplementary CSV保存
        save_enrichment_result_for_supplementary(enrichment_results, output_dir='data/supplementary')
    else:
        print("エンリッチメント結果が得られませんでした")

    
    # # ======= クラスタ4だけ High / Low を比較して保存 =======
    hi_lo_dict = prepare_hi_lo_enrichment(df_all, cluster_id=4,
                                          hi_pct=0.9, lo_pct=0.1, top_n=20)

    create_comparison_bubble_plot(
        hi_lo_dict,
        top_n_per_cluster=13,
        x_label_prefix='',                      # ← 「High / Low」だけ表示
        title='Enrichment Analysis: Very Rapid Cluster Accuracy (High vs Low)',
        save_path='data/data_analyze/enrichment/target_cluster_high_low_enrichment_bubble.pdf'
    )
    # Supplementary CSV保存
    save_enrichment_result_for_supplementary(hi_lo_dict, output_dir='data/supplementary')


if __name__ == "__main__":
    main()