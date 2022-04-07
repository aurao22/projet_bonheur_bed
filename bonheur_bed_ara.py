
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\")
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\ara_commons\\")
from countries.country_constants import get_country_data, get_country_official_name

# ----------------------------------------------------------------------------------
#                        SPECIFIC BONHEUR
# ----------------------------------------------------------------------------------
def get_country_name(current_name, verbose=0):
    # country_code, continent_code, latitude, longitude, a3, official_name, country_id = get_country_data(current_name, verbose=verbose)
    official_name = get_country_official_name(country_name = current_name, verbose=verbose)
    if official_name is None:
        official_name = current_name
    return official_name

def df_remove_duplicated_switch_NA(df, subset=['country_official', 'year'], keep="first", verbose=0):
    df_global_clean = df.copy()
    if verbose:
        print(f"Suppression des lignes dupliquées {df_global_clean[df_global_clean.duplicated(subset=subset)].shape} > {df_global_clean.shape}")
    # Suppression des lignes en doublon
    df_global_clean["NB_NA"] = df_global_clean.isna().sum(axis=1)
    df_global_clean = df_global_clean.sort_values("NB_NA", ascending=True)
    df_global_clean = df_global_clean.drop_duplicates(subset=subset, keep=keep)
    df_global_clean = df_global_clean.drop(columns=["NB_NA"])
    if verbose:
        print(f"{df_global_clean[df_global_clean.duplicated(subset=subset)].shape} > {df_global_clean.shape}")
    return df_global_clean


def complete_df_with_country_datas(df_param, country_col_name="pays", verbose=0):
    # Ajout de la colonne country
    df = None
    if df_param is not None:
        df = df_param.copy()
        df["country_official"] = df[country_col_name].apply(lambda x : get_country_name(x, verbose=verbose))
    return df

def merge_generic_world_data_files(world_datas_files, df_global_completed, data_set_path,country_official_col_name='country_official',country_col_name='country',world_start_with="world_" , verbose=0):
    proceed = set()
    not_proceed = set(world_datas_files.copy())

    dic_world_df = {}
    for file_name in world_datas_files:
        try:
            df_temp = pd.read_csv(data_set_path+file_name, sep=',')
            if verbose:
                print(f"{file_name}==>{df_temp.shape} : ", end="")
                if verbose>1:
                    print(f"{list(df_temp.columns)}")
            prefix = file_name.replace(world_start_with, "").replace(".csv", "")
            to_drop_cols = set()
            cols = list(df_temp.columns)

            if "gini" in prefix.lower() or "Intentional homicide victims".lower() in prefix.lower() or "Subgroup" in cols or "Subgroup".lower() in cols:
                not_proceed.add(file_name)
                if verbose:
                    print("NOT PROCEED")
            else:
                merged_cols = [country_official_col_name]
                for c in cols:
                    # Country or Area	Year	Area	Sex	Record Type	Reliability	Source Year	Value
                    if "Country or Area".lower() == c.lower() or "Country or Territory".lower() == c.lower():
                        df_temp = df_temp.rename(columns={c:country_col_name})
                    else:
                        if "year" == c.lower():
                            df_temp = df_temp.rename(columns={c:c.lower()})
                            merged_cols.append(c.lower())
                        elif "value" == c.lower() or "Annual".lower() == c.lower():  
                            try:
                                df_temp.loc[df_temp[c]=="-9999.9", c] = np.nan
                            except:
                                pass
                            try:
                                df_temp.loc[df_temp[c]==-9999.9, c] = np.nan
                            except:
                                pass
                            df_temp = df_temp.rename(columns={c:prefix})
                        elif "Annual NCDC Computed Value".lower() == c.lower(): 
                            try:
                                df_temp.loc[df_temp[c]=="-9999.9", c] = np.nan
                            except:
                                pass
                            try:
                                df_temp.loc[df_temp[c]==-9999.9, c] = np.nan
                            except:
                                pass 
                            df_temp = df_temp.rename(columns={c:prefix+" NCDC Computed"})
                        else:
                            to_drop_cols.add(c)

                try:
                    # On ne garde que les lignes de total
                    df_temp = df_temp[df_temp["Area"] == "Total"]
                except :
                    pass

                try:
                    v = list(df_temp["Sex"].unique())
                    if "Both Sexes" in v:
                        df_temp = df_temp[df_temp["Sex"] == "Both Sexes"]
                except:
                    pass

                if verbose>1:
                    print(f"{file_name}==>{df_temp.shape} : {list(df_temp.columns)}")
                df_temp = complete_df_with_country_datas(df_temp[df_temp[country_col_name].notna()], country_col_name=country_col_name, verbose=0)
                
                if len(world_datas_files)>1:
                    to_drop_cols.add(country_col_name)

                if verbose>1:
                    print(f"{file_name}==>{df_temp.shape} : {list(df_temp.columns)}")
                    print(f"{file_name}==>Suppression des colonnes inutiles :{to_drop_cols}")
                
                for c in to_drop_cols:
                    df_temp = df_temp.drop(c, axis=1)

                # Suppression des lignes en doublon
                df_temp = df_remove_duplicated_switch_NA(df_temp,subset=list(df_temp.columns), keep="first", verbose=verbose-1)
                               
                # Fusion avec la DF Globale
                try:
                    try:
                        df_temp["year"] = df_temp["year"].astype(int)
                    except:
                        pass
                    if verbose:
                        if verbose>1:
                            print(f"{file_name}==>{df_temp.shape} : {list(df_temp.columns)}")
                        print(f"GLOBAL DF ==>{df_global_completed.shape}", end="")
                    df_global_completed = df_global_completed.merge(df_temp, how='left', on=merged_cols, indicator=len(world_datas_files)==1)
                    proceed.add(file_name)

                    df_global_completed = df_remove_duplicated_switch_NA(df_global_completed,subset=['country_official', 'year'], keep="first", verbose=verbose-1)
                    
                    if verbose:
                        print(f" ==>{df_global_completed.shape}")
                except Exception as error:
                    print(f"{file_name}==> MERGE with Global DF FAIL : {error}")
                dic_world_df[file_name] = df_temp
        except Exception as error:
            print(f"{file_name}==>{error}")

    try:
        df_global_completed = df_global_completed.sort_values(['country_official', 'year'])
    except Exception as error:
        print(f"FAIL to convert year to INT : {error}")
    return df_global_completed, dic_world_df, proceed, not_proceed


def load_scores_files(score_dataset_filenames, data_set_path, country_col_name = "country", country_official_name = 'country_official', score_rapport_with="Rapport-bonheur-", verbose=0):
    df_origine = None
    df_origine_by_line = None
    min_year = -1
    max_year = -1
    country_origin_col_name = country_col_name+"_origin"

    sizes = []

    for score_file_name in tqdm(score_dataset_filenames):
        # Récupération de l'année dans le nom du fichier
        year = score_file_name.replace(score_rapport_with,"")
        year = year.split(".")[0].strip()
        
        # chargement des données
        df_temp = pd.read_csv(data_set_path+score_file_name, sep=',')
        df_temp = df_temp.sort_values(by=country_col_name)

        # Suppression des caractères spéciaux dans les noms de pays
        df_temp[country_col_name] = df_temp[country_col_name].str.replace("*", "", regex=False)
        df_temp[country_col_name] = df_temp[country_col_name].str.strip()

        df_temp[country_origin_col_name] = df_temp[country_col_name]
        
        # Correction des pays
        # df_temp.loc[df_temp[country_col_name] == "Taiwan Province of China", country_col_name] = "Republic of China"
        # df_temp.loc[df_temp[country_col_name] == "Trinidad & Tobago", country_col_name] = "Trinidad and Tobago"
        # df_temp.loc[df_temp[country_col_name] == "Hong Kong", country_col_name] = "Hong Kong S.A.R. of China"
        # df_temp.loc[df_temp[country_col_name] == "Eswatini, Kingdom of", country_col_name] = "Eswatini"
        # df_temp.loc[df_temp[country_col_name] == "North Cyprus", country_col_name] = "Northern Cyprus"
        # df_temp.loc[df_temp[country_col_name] == "Czechia", country_col_name] = "Czech Republic"

        # # df_temp.loc[df_temp[country_col_name] == "Congo (Kinshasa)", country_col_name] = "Democratic Republic of the Congo"
        # # df_temp.loc[df_temp[country_col_name] == "Congo (Brazzaville)", country_col_name] = "Republic of the Congo"
        # df_temp.loc[df_temp[country_col_name] == "Swaziland", country_col_name] = "Eswatini"
        df_temp[country_col_name] = df_temp[country_col_name].str.strip()

        # Ajout de la colonne avec le nom officiel du pays
        df_temp = complete_df_with_country_datas(df_temp, country_col_name=country_col_name, verbose=verbose)
        cols_names = list(df_temp.columns)
        cols_names.remove(country_official_name)
        cols_names.remove(country_col_name)
        cols_names.insert(0, country_official_name)
        cols_names.insert(1, country_col_name)
        df_temp = df_temp[cols_names]
        try:
            df_temp = df_temp.sort_values(by=["rank"])
        except:
            pass
        
        # Il faut traiter les doublons dès maintenant pour éviter des soucis dans la suite des traitements, donc pour les pays : `Republic of Cyprus` et `Kingdom of Sweden`
        previous_size = df_temp.shape[0]
        if verbose:
            print(df_temp.duplicated(subset=[country_official_name]))
        df_temp = df_temp.drop_duplicates(subset=[country_official_name], keep="first")
        if verbose:
            print(f"{score_file_name} before drop duplicated on {country_official_name} : {previous_size}, AFTER : {df_temp.shape[0]} => {previous_size-df_temp.shape[0]} rows DELETED")

        df_temp2 = df_temp.copy()

        initial_columns_name = list(df_temp.columns)

        for col in initial_columns_name:
            # Suppression des colonnes non utilisées
            if "Explained by:" in col or "ladder" in col.lower():
                df_temp = df_temp.drop(col, axis=1)
                df_temp2 = df_temp2.drop(col, axis=1)
            else:
                # Uniformisation des noms de colonne
                if "residual" in col:
                    df_temp = df_temp.rename(columns={col:'Dystopia + residual'})
                    df_temp2 = df_temp2.rename(columns={col:'Dystopia + residual'})
                    col = 'Dystopia + residual'
                elif "whisker"  in col.lower() and "low"  in col.lower():
                    df_temp = df_temp.rename(columns={col:'Whisker-low'})
                    df_temp2 = df_temp2.rename(columns={col:'Whisker-low'})
                    col = 'Whisker-low'
                elif "whisker"  in col.lower() and "upper"  in col.lower():
                    df_temp = df_temp.rename(columns={col:'Whisker-high'})
                    df_temp2 = df_temp2.rename(columns={col:'Whisker-high'})
                    col = 'Whisker-high'

                # Ajout de l'année dans les noms de colonnes ou à l'inverse suppression de l'année
                if year not in col and country_official_name != col:
                    df_temp = df_temp.rename(columns={col:year+"-"+col})
                elif col != year:
                    df_temp2 = df_temp2.rename(columns={col:col.replace(year+"-", "")})
                elif col == year:
                    df_temp2 = df_temp2.rename(columns={col:"score"})
        
        # Ajout de la variable année pour la DF en ligne
        df_temp2["year"] = year
        if df_origine is None:
            df_origine = df_temp
            df_origine_by_line = df_temp2
            min_year = int(year)
            max_year = int(year)
            
        else:
            if int(year) > max_year:
                max_year = int(year)
            if int(year) < min_year:
                min_year = int(year)
            # df_origine = df_origine.merge(df_temp, on=country_col_name, how="outer", indicator=True)
            df_origine = df_origine.merge(df_temp, on=country_official_name, how="outer", indicator=True)
            df_origine = df_origine.rename(columns={"_merge":year+"_merge"})
            df_origine_by_line = pd.concat([df_origine_by_line,df_temp2], axis=0)
            
        sizes.append(f"{score_file_name} CURRENT : {df_temp.shape}  ==> : {df_origine.shape} ==> {df_origine_by_line.shape}")

    df_origine_by_line = df_origine_by_line.sort_values(by=country_official_name)
    df_origine = df_origine.sort_values(by=country_official_name)

    if verbose:
        for line in sizes:
            print(f"{line}")
 
    df_origine_light = df_origine.copy()

    # On complète les données
    df_origine_light["2020-Regional indicator"] = df_origine_light["2020-Regional indicator"].fillna(df_origine_light["2021-Regional indicator"])
    df_origine_light["2021-Regional indicator"] = df_origine_light["2021-Regional indicator"].fillna(df_origine_light["2020-Regional indicator"])
    df_origine_light = df_origine_light.rename(columns={"2020-Regional indicator":"Regional indicator"})
    df_origine_light = df_origine_light.drop("2021-Regional indicator", axis=1)

    # Suppression des colonnes devenues inutiles
    for i in range (min_year, max_year+1, 1):
        try:
            c_n = str(i)
            df_origine_light = df_origine_light.drop(c_n+"_merge", axis=1)
        except:
            pass
    # correction des types des colonnes
    df_origine_light = df_correct_type_to_float(df_origine_light, exclude_cols=[country_col_name, country_official_name, "Regional indicator"], verbose=verbose)
    df_origine = df_correct_type_to_float(df_origine, exclude_cols=[country_col_name, country_official_name, "Regional indicator"], verbose=verbose)
    df_origine_by_line = df_correct_type_to_float(df_origine_by_line, exclude_cols=[country_col_name, country_official_name, "Regional indicator"], verbose=verbose)
            
    # Réorganisation des colonnes
    cols = list(df_origine_light.columns)
    
    cols.remove(country_official_name)
    cols.insert(0, country_official_name)
    cols.remove("Regional indicator")
    cols.insert(1, "Regional indicator")
    current = 2
    # Suppression des colonnes devenues inutiles
    for i in range (min_year, max_year+1, 1):
        try:
            c_n = str(i)
            cols.remove(c_n)
            cols.remove(c_n+"-"+country_origin_col_name)
            cols.insert(current, c_n+"-"+country_origin_col_name)
            cols.insert(current+3, c_n)
            current += 1
        except:
            pass
    
    df_origine_light = df_origine_light[cols]

    return df_origine_light, df_origine_by_line, df_origine


def score_by_line_merge_official_historic(df_official_historic, df_score_by_line, verbose=0):
    # Concaténation des DF pour avoir une seule DF finale
    df_light_completed_by_line_v1 = pd.concat([df_official_historic, df_score_by_line])
    df_light_completed_by_line_v1 = df_light_completed_by_line_v1.sort_values(["country_official", "year"])

    if verbose:
        print(f"Suppression des NA sur le score : {df_light_completed_by_line_v1.shape}", end="")
    df_light_completed_by_line_v1 = df_light_completed_by_line_v1[df_light_completed_by_line_v1["score"].notna()]
    if verbose:
        print(f" => {df_light_completed_by_line_v1.shape}")

    if verbose:
        print(f"Suppression des lignes dupliquées : {df_light_completed_by_line_v1.shape}", end="")
    df_light_completed_by_line_v1["NB_NA"] = df_light_completed_by_line_v1.isna().sum(axis=1)
    df_light_completed_by_line_v1 = df_light_completed_by_line_v1.sort_values("NB_NA", ascending=True)
    df_light_completed_by_line_v1 = df_light_completed_by_line_v1.drop_duplicates(subset=['country_official', 'year'], keep="first")
    df_light_completed_by_line_v1 = df_light_completed_by_line_v1.drop(columns=["NB_NA"])
    
    if verbose:
        print(f" => {df_light_completed_by_line_v1.shape}")

    # Réorganisation des données
    df_light_completed_by_line_v1 = score_by_line_clean_index_and_sort(df_light_completed_by_line_v1, verbose=verbose)
    return df_light_completed_by_line_v1

def score_by_line_clean_index_and_sort(df, verbose=0):
    # Réorganisation des données
    if verbose:
        print(f"Trie des données", end="")
    df = df.sort_values(["country_official", "year"])
    df = df.reset_index()
    df = df.drop("index", axis=1)
    if verbose:
        print(f".... END")
    return df


def score_by_line_complete_with_historic_score(df_evolution_orgin, df_origine_by_line, verbose=0):
    # copie des données initiales
    if verbose:
        print("INPUT",df_origine_by_line.shape," dont score NA :", df_origine_by_line["score"].isna().sum())
    df_light_completed_by_line = df_origine_by_line[df_origine_by_line["score"].notna()].copy()
    df_evolution_orgin_by_line = df_evolution_orgin.copy()

    # Ajout des données du fichier historique des scores
    for i in range (2015, 2021):
        # Création d'une DF temporaire pour inverser les valeurs
        df_temp = df_evolution_orgin_by_line[['country', 'country_official', 'score_'+str(i)]].copy()
        df_temp = df_temp[df_temp['score_'+str(i)].notna()]
        df_temp["year"] = i
        df_temp = df_temp.rename(columns={'score_'+str(i):"score"})
        # on supprime les valeurs na
        if verbose>1:
            print(i, "score NA :", df_temp["score"].isna().sum(), end="")
        df_temp = df_temp[df_temp["score"].notna()]
        if verbose>1:
            print(" - AFTER NA :", df_temp["score"].isna().sum())

        # Concaténation des DF pour avoir une seule DF finale
        df_light_completed_by_line = pd.concat([df_light_completed_by_line, df_temp])

    if verbose:
        print("OUTPUT",df_light_completed_by_line.shape," dont score NA :", df_light_completed_by_line["score"].isna().sum())
    return df_light_completed_by_line

def fill_na_regional_indicator(df_param, verbose=0):

    # copie des données initiales
    df = df_param.copy()
    regions_group = df[df['Regional indicator'].notna()].groupby(['country_official', 'Regional indicator']).agg({'Regional indicator':['count']})
    regions_group = regions_group.reset_index()
    regions_group.columns = regions_group.columns.droplevel()
    regions_group.columns = ['country_official', 'Regional indicator2',"count Regional indicator"]
    regions_group = regions_group.sort_values(by=['count Regional indicator'], ascending=False)
    
    if verbose>1:
        print("nb_regions : ", regions_group.shape, end="")

    regions_group = regions_group.drop_duplicates(subset=["country_official"], keep="first")
    regions_group = regions_group.drop("count Regional indicator", axis=1)

    if verbose>1:
        print(" after drop duplicated : ", regions_group.shape)

    # Fusion des DF
    df = df.merge(regions_group, on="country_official", how="left", indicator=False)

    if verbose:
        print("INPUT Regional indicator NA : ",df["Regional indicator"].isna().sum(), end="")
    df["Regional indicator"] = df["Regional indicator"].fillna(df["Regional indicator2"])    
    if verbose:
        print(" => OUTPUT : ",df["Regional indicator"].isna().sum())

    # Suppression de la colonne ajoutée
    df = df.drop(["Regional indicator2"], axis=1)
    return df

def merge_and_clean_country_by_year(df, start=2019, end=2022, verbose=0):
    nb_years = end - start
    df_origine_light_completed2 = df.copy()
    df_origine_light_completed2 = df_origine_light_completed2.rename(columns={"2019-country_origin":"country_origin", "2019-country":"country"})

    print(f"country_origin na:{df_origine_light_completed2['country_origin'].isna().sum()} and country na:{df_origine_light_completed2['country'].isna().sum()}")
    for i in range(start+1, end+1, 1):
        df_origine_light_completed2["country_origin"] = df_origine_light_completed2["country_origin"].fillna(df_origine_light_completed2[str(i)+"-country_origin"])
        df_origine_light_completed2["country"] = df_origine_light_completed2["country"].fillna(df_origine_light_completed2[str(i)+"-country"])
        cn = str(i)
        try:
            df_origine_light_completed2 = df_origine_light_completed2.drop(cn+"-country_origin", axis=1)
            df_origine_light_completed2 = df_origine_light_completed2.drop(cn+"-country", axis=1)
        except:
            pass
        print(f"country_origin na:{df_origine_light_completed2['country_origin'].isna().sum()} and country na:{df_origine_light_completed2['country'].isna().sum()}")
    try:
        # On ne garde que le pays d'origine
        df_origine_light_completed2 = df_origine_light_completed2.drop("country", axis=1)
    except:
        pass

    cols = list(df_origine_light_completed2.columns)
    cols.remove("country_origin")
    cols.insert(1, "country_origin")

    init_pos = 3
    for i in tqdm(range(start, end+1, 1)):
        cn = str(i)
        cols.remove(cn)
        cols.insert(init_pos, cn)
        try:
            cols.remove(cn+"-rank")
            cols.insert(init_pos+nb_years, cn+"-rank")
        except:
            pass
        cols.remove(cn+"-PIB")
        cols.insert(init_pos+(nb_years*2), cn+"-PIB")
        init_pos +=1
        
    df_origine_light_completed2 = df_origine_light_completed2[cols]
    return df_origine_light_completed2


def df_correct_type_to_float(df_param, rounded=3, exclude_cols=[], verbose=0):
    # correction des types des colonnes
    df = df_param.copy()
    cols = list(df.columns)
    for c in exclude_cols:
        try:
            cols.remove(c)
        except:
            pass
    
    for c_n in cols:
        try:
            df[c_n] = df[c_n].str.replace(",", ".", regex=False)
            df[c_n] = df[c_n].astype(float)            
            if rounded is not None:
                df[c_n] = df[c_n].apply(lambda x: round(x, rounded))
        except:
            try:
                df[c_n] = df[c_n].str.replace(".", ",", regex=False)
            except:
                pass
    return df

# ----------------------------------------------------------------------------------
#                        GENERIC FUNCTIONS
# ----------------------------------------------------------------------------------

def get_dir_files(dir_path, start_with=None, endwith=None, verbose=0):
    fichiers = None
    if endwith is not None:
        if start_with is not None:
            fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(endwith) and f.startswith(start_with)]
        else:
            fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(endwith)]
    else:
        if start_with is not None:
            fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.startswith(start_with)]
        else:
            fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return fichiers


def get_numeric_columns_names(df, verbose=False):
    """Retourne les noms des colonnes numériques
    Args:
        df (DataFrame): Données
        verbose (bool, optional): Mode debug. Defaults to False.

    Returns:
        List(String): liste des noms de colonne
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    return list(newdf.columns)


def get_outliers_datas(df, colname):
    """[summary]

    Args:
        df ([type]): [description]
        colname ([type]): [description]

    Returns:
        (float, float, float, float): q_low, q_hi,iqr, q_min, q_max
    """
    # .quantile(0.25) pour Q1
    q_low = df[colname].quantile(0.25)
    #  .quantité(0.75) pour Q3
    q_hi  = df[colname].quantile(0.75)
    # IQR = Q3 - Q1
    iqr = q_hi - q_low
    # Max = Q3 + (1.5 * IQR)
    q_max = q_hi + (1.5 * iqr)
    # Min = Q1 - (1.5 * IQR)
    q_min = q_low - (1.5 * iqr)
    return q_low, q_hi,iqr, q_min, q_max


# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR


def color_graph_background(ligne=1, colonne=1):
    figure, axes = plt.subplots(ligne,colonne)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    if isinstance(axes, np.ndarray):
        for axe in axes:
            # Traitement des figures avec plusieurs lignes
            if isinstance(axe, np.ndarray):
                for ae in axe:
                    ae.set_facecolor(PLOT_BAGROUNG_COLOR)
            else:
                axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    else:
        axes.set_facecolor(PLOT_BAGROUNG_COLOR)
    return figure, axes


def graphe_outliers(df_out, column, q_min, q_max):
    """[summary]

    Args:
        df_out ([type]): [description]
        column ([type]): [description]
        q_min ([type]): [description]
        q_max ([type]): [description]
    """
    
    figure, axes = color_graph_background(1,2)
    # Avant traitement des outliers
    # Boite à moustaches
    #sns.boxplot(data=df_out[column],x=df_out[column], ax=axes[0])
    df_out.boxplot(column=[column], grid=True, ax=axes[0])
    # scatter
    df_only_ok = df_out[(df_out[column]>=q_min) & (df_out[column]<=q_max)]
    df_only_ouliers = df_out[(df_out[column]<q_min) | (df_out[column]>q_max)]
    plt.scatter(df_only_ok[column].index, df_only_ok[column].values, c='blue')
    plt.scatter(df_only_ouliers[column].index, df_only_ouliers[column].values, c='red')
    # Dimensionnement du graphe
    figure.set_size_inches(18, 7, forward=True)
    figure.set_dpi(100)
    figure.suptitle(column, fontsize=16)
    plt.show()


def draw_correlation_graphe(df, title, verbose=False, annot=True, fontsize=5):
    """Dessine le graphe de corrélation des données

    Args:
        df (DataFrame): Données à représenter
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    corr_df = df.corr()
    if verbose:
        print("CORR ------------------")
        print(corr_df, "\n")
    figure, ax = color_graph_background(1,1)
    figure.set_size_inches(18, 15, forward=True)
    figure.set_dpi(100)
    figure.suptitle(title, fontsize=16)
    sns.heatmap(corr_df, annot=annot, annot_kws={"fontsize":fontsize})
    plt.xticks(rotation=45, ha="right", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    # Bolivie

    pays = ['Bolivie', 'Congo (Kinshasa)', 'Congo (RDC)', 'Eswatini', 'Iran',
       'Laos', 'Moldavie', 'Corée du Sud', 'Palestine', 'Syrie', 'Taïwan',
       'Tanzanie', 'Venezuela', 'Viêt Nam']
    for p in pays:
        print(get_country_name(current_name=p, verbose=1))
    print("END")


    pays = ["Bolivia","Czech Republic","Iran","Kosovo","Laos","Moldova","Somaliland region","South Korea","Syria","Taiwan Province of China","Tanzania","Venezuela","Vietnam"]
    for p in pays:
        print(get_country_name(current_name=p, verbose=1))
    print("END")