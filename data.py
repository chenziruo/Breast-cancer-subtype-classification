import pandas as pd
# 读取 TCGA 标本表与 subtype 原表
samples = pd.read_csv("E:/De/bioo/data/biospecimen.project-tcga-brca/sample.tsv", sep="\t", dtype=str)
subtypes = pd.read_csv("E:/De/bioo/data/metadata_subtypes_raw.csv", dtype=str)

# 取出 sample submitter id (sample barcode) 与 patient id
map_df = samples[['cases.submitter_id','samples.submitter_id']].dropna()
map_df.columns = ['patient','sample_barcode']

# 截取 patient 前12位确保格式一致（若必要）
map_df['patient12'] = map_df['patient'].str[:12]

# 取 subtype 表中 patient->PAM50 映射
sub = subtypes[['patient','BRCA_Subtype_PAM50']].dropna()
sub['patient12'] = sub['patient'].str[:12]

# 合并，得到每个 sample_barcode 的 Subtype（可能有 NA）
merged = map_df.merge(sub[['patient12','BRCA_Subtype_PAM50']], on='patient12', how='left')
merged = merged[['sample_barcode','BRCA_Subtype_PAM50']].rename(columns={'sample_barcode':'sample','BRCA_Subtype_PAM50':'Subtype'})

# 保存为 pipeline 可用格式（index为样本条码或保留为列）
merged.to_csv("E:/De/bioo/data/metadata.csv", index=False)