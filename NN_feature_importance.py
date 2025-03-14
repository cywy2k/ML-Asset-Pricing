# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

# %%
for i in range(1,6):
    df_import = pd.read_parquet(f"C:/Users/CYWY2K/HKUST/STUDY/PG_1/MAFS5440/PROJECT-2/model/nn_top_results/top_Importence_NN{i}.parquet")
    

    df_import.index = df_import["Feature"]

    df_import.index = df_import.index.str.split('_').str[0]

    df = df_import.groupby(df_import.index).sum()
    df["Feature"] = df.index
    df.sort_values(by="Importance",inplace=True,ascending=False)

    print(df["Importance"].sum())
    top_features_df = df.head(20)
    print(df)

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
    plt.title(f'Top 20 Feature Importances for NN{i}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    output_path = f"feature_importances_top_20_NN{i}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()


