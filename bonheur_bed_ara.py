
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sys
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\")
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\ara_commons\\")
from ara_commons.countries.country_constants import get_country_data, get_country_official_name
from ara_commons.ara_df import remove_na_columns

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
            cols = list(df_temp.columns)

            if "gini" in prefix.lower() or "Intentional homicide victims".lower() in prefix.lower() or ("Subgroup" in cols and not "Women_Par_100_Men".lower() in prefix.lower()) or "Subgroup".lower() in cols:
                if verbose:
                    print("NOT PROCEED")
            else:        
                # Fusion avec la DF Globale
                try:
                    df_global_completed, df_temp = merge_generic_world_df(df_temp, df_global_completed, prefix,country_official_col_name=country_official_col_name,country_col_name=country_col_name, verbose=verbose)
                    proceed.add(file_name)
                    if verbose:
                        print(f" ==>{df_global_completed.shape}")
                except Exception as error:
                    print(f"{file_name}==> MERGE with Global DF FAIL : {error}")
                dic_world_df[file_name] = df_temp
        except Exception as error:
            print(f"{file_name}==>{error}")
    
    df_global_completed = df_global_completed.sort_values(['country_official', 'year'])
    
    for tp in proceed:
        not_proceed.remove(tp)

    return df_global_completed, dic_world_df, proceed, not_proceed


def merge_generic_world_data_files_save(world_datas_files, df_global_completed, data_set_path,country_official_col_name='country_official',country_col_name='country',world_start_with="world_" , verbose=0):
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
    for tp in proceed:
        not_proceed.remove(tp)
    return df_global_completed, dic_world_df, proceed, not_proceed


def merge_generic_world_df(df_temp, df_global_completed, prefix,country_official_col_name='country_official',country_col_name='country', verbose=0):
                
    to_drop_cols = set()
    cols = list(df_temp.columns)
    
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
        print(f"{prefix}==>{df_temp.shape} : {list(df_temp.columns)}")
    df_temp = complete_df_with_country_datas(df_temp[df_temp[country_col_name].notna()], country_col_name=country_col_name, verbose=0)
    
    to_drop_cols.add(country_col_name)

    if verbose>1:
        print(f"{prefix}==>{df_temp.shape} : {list(df_temp.columns)}")
        print(f"{prefix}==>Suppression des colonnes inutiles :{to_drop_cols}")
    
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
                print(f"{prefix}==>{df_temp.shape} : {list(df_temp.columns)}")
            print(f"GLOBAL DF ==>{df_global_completed.shape}", end="")
        df_global_completed = df_global_completed.merge(df_temp, how='left', on=merged_cols, indicator=False)
        df_global_completed = df_remove_duplicated_switch_NA(df_global_completed,subset=['country_official', 'year'], keep="first", verbose=verbose-1)
        
        if verbose:
            print(f" ==>{df_global_completed.shape}")
    except Exception as error:
        print(f"{prefix}==> MERGE with Global DF FAIL : {error}")
    
    return df_global_completed, df_temp


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

def load_world_gini_file(data_set_path, file_name, df,subset=['country_official', 'year'], excluded_cols=[], indicator=False, verbose=0):
    gini_file_path = data_set_path + file_name
    
    df_gini = pd.read_csv(gini_file_path, sep=',')
    df_gini = remove_na_columns(df_gini, max_na=85, excluded_cols=excluded_cols, verbose=verbose-1, inplace=False)
    df_gini = df_gini[df_gini['Country Name'].notna()]
    if verbose:
        print(f"{df_gini.shape} données chargées ------> {list(df_gini.columns)}")
    df_gini = complete_df_with_country_datas(df_gini, 'Country Name', verbose=verbose-1)
    df_gini = df_gini.drop(['Country Name','Indicator Name', 'Indicator Code'], axis=1)
    
    # Préparation de l'ajout des données
    years_col_names = list(df_gini.columns)
    to_keep_cols = ['country_official', 'Country Code']

    for c in to_keep_cols:
        years_col_names.remove(c)

    # Construction de la seconde DF
    df_gini2 = None
    for y in years_col_names:
        y_cols = to_keep_cols.copy()
        y_cols.append(y)
        df_temp = df_gini[y_cols].copy()
        df_temp["year"] = y
        df_temp = df_temp.rename(columns= {y:"gini"})
        df_temp = df_remove_duplicated_switch_NA(df_temp, verbose=verbose-1)
        try:
            df_temp["year"] = df_temp["year"].astype(int)
        except:
            pass
        if df_gini2 is None:
            df_gini2 = df_temp
        else:
            df_gini2 = pd.concat([df_gini2,df_temp], axis=0)
            df_gini2 = df_remove_duplicated_switch_NA(df_gini2,subset=subset, keep="first", verbose=verbose-1)
    try:
        df_gini2 = df_correct_type_to_float(df_gini2, rounded=3, exclude_cols=[], verbose=verbose-1)
    except:
        pass
    res =  df.merge(df_gini2, how='left', on=subset, indicator=indicator)
    res = df_remove_duplicated_switch_NA(res,subset=subset, keep="first", verbose=verbose-1)
                    
    return res, df_gini2


def load_world_homicide_file(file_path, file_name, df,subset=['country_official', 'year'], excluded_cols=[], indicator=False, verbose=0):

    unit="rate"
    if unit.lower() not in file_name.lower():
        unit = "nb"
        
    df_origin = pd.read_csv(file_path+file_name, sep=',')
    df_origin = remove_na_columns(df_origin, max_na=85, excluded_cols=excluded_cols, verbose=verbose-1, inplace=False)
    df_origin = df_origin[df_origin['Country'].notna()]
    if verbose:
        print(f"{df_origin.shape} données chargées ------> {list(df_origin.columns)}")
    df_origin = complete_df_with_country_datas(df_origin, 'Country', verbose=verbose-1)
    df_origin = df_origin.drop(['Region', 'Subregion', 'Country','Source'], axis=1)

    # Préparation de l'ajout des données
    years_col_names = list(df_origin.columns)
    to_keep_cols = ['country_official']
    for c in to_keep_cols:
        try:
            years_col_names.remove(c)
        except:
            pass

    df_gender = None
    sub_df = []
    genders_cols_names = []
    for gender in df_origin["Gender"].unique():
        selection = df_origin[df_origin["Gender"]==gender].copy()
        for y in years_col_names:
            if "Nb_".lower() not in y.lower() and "Gender".lower() not in y.lower():
                y_cols = to_keep_cols.copy()
                y_cols.append(y)
                df_temp = selection[y_cols].copy()
                df_temp["year"] = y
                new_name = "intentional homicide victims "+gender+" "+unit
                df_temp = df_temp.rename(columns= {y:new_name})
                genders_cols_names.append(new_name)
                df_temp = df_remove_duplicated_switch_NA(df_temp, verbose=verbose-1)
                try:
                    df_temp["year"] = df_temp["year"].astype(int)
                except:
                    pass
                if df_gender is None:
                    df_gender = df_temp
                else:
                    df_gender = pd.concat([df_gender,df_temp], axis=0)
                    # df_homicide = df_remove_duplicated_switch_NA(df_homicide,subset=subset, keep="first", verbose=verbose-1)
        sub_df.append(df_gender)
        df_gender = None

    df_homicide = None
    for df_gender in sub_df:
        if df_homicide is None:
            df_homicide = df_gender
        else:
            df_homicide = df_homicide.merge(df_gender, how='outer', on=subset)

    try:
        df_homicide = df_correct_type_to_float(df_homicide, rounded=3, exclude_cols=[], verbose=verbose-1)
    except:
        pass
    res =  df.merge(df_homicide, how='left', on=subset, indicator=indicator)
    res = df_remove_duplicated_switch_NA(res,subset=subset, keep="first", verbose=verbose-1)

    # on ne fait la somme que pour les nombres, car le ratio est la part des homicides par sexe et non total pays
    if "nb" in unit.lower():
        res["Homicide victime "+unit] = 0
        for g in genders_cols_names:
            res["Homicide victime "+unit] = res["Homicide victime "+unit] + res[g]
    
    return res, df_homicide

def load_world_population_file(file_path, file_name, df,subset=['country_official', 'year'],excluded_cols=[], indicator=False, verbose=0):

    df_origin = pd.read_csv(file_path+file_name, sep=',')
    df_origin = remove_na_columns(df_origin, max_na=85, excluded_cols=excluded_cols, verbose=verbose-1, inplace=False)
    df_origin = df_origin[df_origin['Country or Area'].notna()]
    try:
        df_origin = df_origin[df_origin['Subgroup'].notna()]
    except:
        pass

    if verbose:
        print(f"{df_origin.shape} données chargées ------> {list(df_origin.columns)}")

    sub_df = []
    genders = list(df_origin["Subgroup"].unique())
    df_global_completed = df.copy()
    genders_cols_names = []
    for gender in genders:
        df_global_completed, df_temp = merge_generic_world_df(df_origin[df_origin["Subgroup"]==gender], df_global_completed, prefix=gender+" Population",country_official_col_name='country_official',country_col_name='country', verbose=verbose)    
        sub_df.append(df_temp)
        genders_cols_names.append(gender+" Population")

    df_population = None
    for df_gender in sub_df:
        if df_population is None:
            df_population = df_gender
        else:
            df_population = df_population.merge(df_gender, how='outer', on=subset)
   
    res = df_remove_duplicated_switch_NA(df_global_completed,subset=subset, keep="first", verbose=verbose-1)
    
    # Trouver la chaine commune entre 2 strings
    common = ""
    try:
        common = genders_cols_names[0]
        end = False
        while not end:
            if common in genders_cols_names[1]:
                end = True
                common = common.strip()
            else:
                if len(common)>1:
                    common = common[0:-2]
                else:
                    common = ""
                    end = True
    except:
        common = ""
    
    res[common+"Population"] = 0
    for g in genders_cols_names:
        res[common+"Population"] = res[common+"Population"] + res[g]

    return res, df_population

def load_world_unemployment_file(file_path, file_name, df,subset=['country_official', 'year'],excluded_cols=[], indicator=False, verbose=0):

    df_origin = pd.read_csv(file_path+file_name, sep=',')
    if verbose:
        print(f"{df_origin.shape} données chargées ------> {list(df_origin.columns)}")
    
    df_origin = remove_na_columns(df_origin, max_na=85,excluded_cols=excluded_cols, verbose=verbose-1, inplace=False)
    if verbose: print('NA columns removed', df_origin.shape)
    df_origin = df_origin[df_origin['Country or Area'].notna()]
    if verbose: print('Country or Area NA removed', df_origin.shape)
    df_origin = df_origin[df_origin['Subgroup'].notna()]
    if verbose: print('Subgroup NA removed', df_origin.shape)
    df_origin = df_origin[~df_origin['Subgroup'].str.contains("-24 yr")]
    if verbose: print('Subgroup -24 yr removed', df_origin.shape)

    sub_df = []
    genders = list(df_origin["Subgroup"].unique())
    df_global_completed = df.copy()
    genders_cols_names = []
    for gender in genders:
        df_global_completed, df_temp = merge_generic_world_df(df_origin[df_origin["Subgroup"]==gender], df_global_completed, prefix=gender+" Unemployment rate",country_official_col_name='country_official',country_col_name='country', verbose=verbose)    
        genders_cols_names.append(gender+" Unemployment rate")
        sub_df.append(df_temp)

    df_population = None
    for df_gender in sub_df:
        if df_population is None:
            df_population = df_gender
        else:
            df_population = df_population.merge(df_gender, how='outer', on=subset)
   
    res = df_remove_duplicated_switch_NA(df_global_completed,subset=subset, keep="first", verbose=verbose-1)
    
    res["Unemployment rate"] = 0
    for g in genders_cols_names:
        res["Unemployment rate"] = res["Unemployment rate"] + res[g]
    return res, df_population


def get_year_data_mean(df, current_y, current_country, current_data_col_name, verbose=0):
    val = df.loc[(df['year']==current_y) & (df['country_official']==current_country),current_data_col_name].values[0]
    if np.isnan(val) or val == 0:
        c_y_min = np.min(df.loc[df['country_official']==current_country,"year"])
        c_y_max = np.max(df.loc[df['country_official']==current_country,"year"])

        y_down = current_y-1
        y_up = current_y+1

        if c_y_min > y_down:
            # on est déjà à la date minimale, donc on prend la date suivante +1
            y_down = current_y+2

        if c_y_max < y_up:
            # on est déjà à la date maximale, donc on prend la date précédente -1
            y_up = current_y-2

        year_moins = np.nan
        try:
            year_moins = df.loc[(df['year']==y_down) & (df['country_official']==current_country),current_data_col_name].values[0]
        except Exception as error:
            if verbose>1:
                print(f"{current_country} - {current_y} - {current_data_col_name} Exception No year less for ({y_down} : {error}")
            year_moins = np.nan

        year_plus = np.nan
        try:
            year_plus = df.loc[(df['year']==y_up) & (df['country_official']==current_country),current_data_col_name].values[0]
        except Exception as error:
            if verbose>1:
                print(f"{current_country} - {current_y} - {current_data_col_name} Exception No year plus for ({y_up} : {error}")
            year_plus = np.nan
        
        if np.isnan(year_plus) and np.isnan(year_moins):
            if verbose>1:
                print(f"{current_country} - {current_y} - {current_data_col_name} - no data for {y_down} and {y_up}")
            try:
                val = df.loc[(df['country_official']==current_country),current_data_col_name].mean()
                if verbose:
                    print(f"{current_country} - {current_y} - {current_data_col_name} new value mean of all datas: {val}")
            except Exception as error:
                if verbose:
                    print(f"{current_country} - {current_y} - {current_data_col_name} Exception on mean({year_plus} and {year_moins}) : {error}")                   
        else:
            try:
                val = np.mean(year_moins, year_plus)
                if verbose:
                    print(f"{current_country} - {current_y} - {current_data_col_name} new value mean previous and next year: {val}")
            except Exception as error:
                if verbose:
                    print(f"{current_country} - {current_y} - {current_data_col_name} Exception on mean({year_plus} and {year_moins}) : {error}")
    else:
        if verbose:
            print(f"{current_country} - {current_y} - {current_data_col_name} value : {val} (no change)")
    return val

def get_df_for_country_data(df, country_official_name):
    country_datas_df = df[df['country_official']==country_official_name].copy()
    country_datas_df = country_datas_df.drop(['continent_encode','continent_code_AF', 'continent_code_AS', 'continent_code_EU',
       'continent_code_NA', 'continent_code_OC', 'continent_code_SA'], axis=1)
    country_datas_df = country_datas_df.set_index('year')
    return country_datas_df


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


def process_one_hot(df, col="description", verbose=0):
    encoder = OneHotEncoder(sparse=False)
    transformed = encoder.fit_transform(df[[col]])
    if verbose:
        print(transformed)
    #Create a Pandas DataFrame of the hot encoded column
    ohe_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out())
    if verbose:
        print("ohe_df:", ohe_df.shape, "data:", df.shape)

    #concat with original data
    df_completed = df.copy()
    df_completed = pd.concat([df_completed, ohe_df], axis=1)
    if verbose:
        print("ohe_df:", ohe_df.shape, "data:", df.shape, "data_encode:", df_completed.shape)
    return df_completed


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


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(18,7), facecolor=PLOT_FIGURE_BAGROUNG_COLOR)
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

acp_colors = {}
def display_factorial_planes(X_projected, centres_reduced_df, n_comp, pca, axis_ranks, labels=None, alpha=0.5, illustrative_var=None, ax=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
                    
            # affichage des points
            if illustrative_var is None:
                ax.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    ax.scatter(X_projected.loc[selected, "F"+str(d1+1)], X_projected.loc[selected, "F"+str(d2+1)], alpha=alpha, label=value)
                ax.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    ax.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = round(np.max(np.max(np.abs(X_projected.max())), axis=None) * 1.1)
            ax.set_xticks(range(-boundary,boundary+1))
            ax.set_yticks(range(-boundary,boundary+1))
                    
            # affichage des lignes horizontales et verticales
            ax.plot([-boundary, boundary], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-boundary, boundary], color='grey', ls='--')
            
            ax.scatter(centres_reduced_df["F"+str(d1+1)], centres_reduced_df["F"+str(d2+1)],
                marker='x', s=169, linewidths=3,
                color='k', zorder=10)

            # nom des axes, avec le pourcentage d'inertie expliqué
            ax.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            ax.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            ax.set_title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
    return ax


def display_factorial_planes_save(X_projected, centres_reduced_df, n_comp, pca, axis_ranks, labels=None, alpha=0.5, illustrative_var=None, ax=None):
    axes = []
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure
            if ax is None:
                fig, ax = plt.subplots(figsize=(10,10))
            axes.append(ax)

            # initialisation de la figure       
            plt.figure(figsize=(18,15), facecolor=PLOT_FIGURE_BAGROUNG_COLOR)
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected.loc[selected, "F"+str(d1+1)], X_projected.loc[selected, "F"+str(d2+1)], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.max(np.abs(X_projected.max())), axis=None) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')
            
            plt.scatter(centres_reduced_df["F"+str(d1+1)], centres_reduced_df["F"+str(d2+1)],
                marker='x', s=169, linewidths=3,
                color='k', zorder=10)

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show()

from matplotlib.collections import LineCollection

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None, ax=None):

    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                ax.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
                ax.set_facecolor(PLOT_BAGROUNG_COLOR)
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        ax.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            ax.add_artist(circle)

            # définition des limites du graphique
            ax.set_yticks(range(ymin, ymax+1))
            ax.set_xticks(range(xmin, xmax+1))
            
            # affichage des lignes horizontales et verticales
            ax.plot([-1, 1], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            ax.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            ax.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            ax.set_title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
    return ax

def display_circles_old(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):

    axes = []

    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,10))
            axes.append(ax)
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
                ax.set_facecolor(PLOT_BAGROUNG_COLOR)
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            fig.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            # plt.show(block=False)
    return axes


def display_factorial_planes_by_theme(X_projected,pca, n_comp, axis_ranks, alpha=0.5, illustrative_var=None, by_theme=False):
    axes = []
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
            # affichage des points
            illustrative_var = np.array(illustrative_var)
            valil = np.unique(illustrative_var)

            figure, axes = plt.subplots(2,len(valil)//2)

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            
            # On commence par traiter le NAN pour plus de lisibilité dans le graphe
            value = str(np.nan)
            i = 0
            j = 0
            if value in valil :
                _display_one_scatter(X_projected, pca, axes[i][j], value, d1, d2, alpha,boundary, illustrative_var)
                valil = valil[valil != value]
                j += 1
            
            for value in valil:
                _display_one_scatter(X_projected, pca, axes[i][j], value, d1, d2, alpha,boundary, illustrative_var)
                
                j += 1
                if j > (len(valil)//2):
                    i += 1
                    j = 0
            
            figure.set_size_inches(18.5, 7, forward=True)
            figure.set_dpi(100)
            figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
            figure.suptitle("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def _display_one_scatter(X_projected, pca, axe,value, d1, d2, alpha, boundary, illustrative_var):
    selected = np.where(illustrative_var == value)
    c=acp_colors.get(value, "blue")
    axe.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, c=c, s=100)
    axe.legend()
    # nom des axes, avec le pourcentage d'inertie expliqué
    axe.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
    axe.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

    axe.set_xlim([-boundary,boundary])
    axe.set_ylim([-boundary,boundary])
    # affichage des lignes horizontales et verticales
    axe.plot([-100, 100], [0, 0], color='grey', ls='--')
    axe.plot([0, 0], [-100, 100], color='grey', ls='--')
    axe.set_facecolor(PLOT_BAGROUNG_COLOR)


from pandas.plotting import parallel_coordinates

def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    
    return (colour[0],colour[1],colour[2],alpha)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''
    palette = sns.color_palette("bright", 10)

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])

    # Create the plot
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.7)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20) 

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''
    palette = sns.color_palette("bright", 10)
    # Create the plot
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20) 

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
    corr_df = round(df.corr(), 2)
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


import plotly.graph_objs as go
import plotly.express as px

def draw_top_score(score_only_by_country_t, n_top = 10,markers=True, verbose=0):

    countries_cols = list(score_only_by_country_t.columns)[0:n_top]

    fig = px.line(score_only_by_country_t[countries_cols], markers=markers, title=f"Evolution des scores du top {n_top} (sur la moyenne).")
    fig.update_layout(
        yaxis_title="Score",
        legend_title=f"{n_top} pays avec le meilleur score",
    )
    fig.update_layout(margin=dict(l=10, r=20, t=40, b=20))
    fig.update_layout(xaxis = go.layout.XAxis( tickangle = 45) )
    fig.update_xaxes(dtick=1)
    fig.show()

from plotly.subplots import make_subplots

def draw_top_by_country_score(score_only_by_country_t, n_top = 5, verbose=0):
    
    countries_names = list(score_only_by_country_t.columns)

    sub_plot_title = []
    i = 1
    for tit in countries_names[0:n_top]:
        sub_plot_title.append(str(i) + " - " + tit)
        i += 1

    years = list(score_only_by_country_t.index)
    fig = make_subplots(rows=n_top+1, cols=1, subplot_titles=tuple(sub_plot_title))
    fig.update_annotations(font_size=12)

    for i in range(0, n_top):
        y = score_only_by_country_t[countries_names[i]]
        sub_fig = go.Scatter(x=years, y=y, name = sub_plot_title[i], showlegend=False)
        fig.add_trace(sub_fig, row=i+1, col=1)

    fig.update_layout(height=200*n_top, width=1000, title_text=f"Evolution des scores pour le top {n_top} des pays")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(nticks=10, dtick=0.1)
    fig.show()

from sklearn.preprocessing import StandardScaler

def draw_country_data_evolution(df, country_official_name):
    df_country = get_df_for_country_data(df, country_official_name=country_official_name)
    # Standardisation des features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_country[get_numeric_columns_names(df_country)])
    df_country_std =pd.DataFrame(scaled_features, index=df_country.index, columns=get_numeric_columns_names(df_country))
    fig = px.line(df_country_std, markers=True, title=f"{country_official_name}")
    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Années",
        legend_title="Données",
    )
    fig.update_layout(margin=dict(l=10, r=20, t=40, b=10))
    fig.update_xaxes(dtick=1)
    fig.show()
    return fig, df_country, df_country_std


def draw_kmeans_features_3d(df, features, verbose=0):
    fig = px.scatter_3d(x=df[features[0]], y=df[features[1]], z=df[features[2]], color=df["kmeans_cluster"], title=f'K-Means Clustering pour les features {features}.')
    fig.update_layout(margin=dict(l=10, r=20, t=40, b=10))
    fig.update_layout(
            xaxis_title=features[0],
            yaxis_title=features[1],
        )
    fig.show()

import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def draw_silhouette_curve(df, X, nb_clusters, random_state=42, x_col = (5, "Score"), y_col=(6, "PIB"), verbose=0):
    
    silhouette_n_clusters = []

    for n_clusters in nb_clusters:
        # Entrainement du modèle
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(X)

        # Calcul du score silhouette
        silhouette_scr = round(silhouette_score(X, cluster_labels),2)
        if verbose:
            print(f"{n_clusters} clusters = {silhouette_scr} silhouette_score")

        silhouette_n_clusters.append(silhouette_scr)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # Représentation graphique
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        # Graphique 1
        ax1.set_title(f"The silhouette plot for the various clusters 0 to {n_clusters}.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_scr, color="red", linestyle="--")

        ax1.set_yticks([])  
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Graphique 2
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        
        ax2.scatter(df.iloc[::,x_col[0]], df.iloc[::,y_col[0]], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # centroides
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Visualisation des clusters sur le dataset")
        ax2.set_xlabel(x_col[1])
        ax2.set_ylabel(y_col[1])

        plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
                    fontsize=14, fontweight='bold')
        plt.show()
    
    # Dernier graphe avec la silhouette
    fig = px.line(x=nb_clusters, y=silhouette_n_clusters , markers=True, title=f"Score silhouette par rapport au nombre de clusters")
    fig.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette score",
    )
    fig.update_layout(margin=dict(l=10, r=20, t=40, b=10))
    fig.update_xaxes(dtick=1)
    fig.show()

    return silhouette_n_clusters

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

def draw_kmeans_and_DBSCAN_comparison(X_scale, nb_clusters, features_column_name=["PIB", "Soutien"], verbose=0):
    
    fte_colors = {
            0: "#008fd5",
            1: "#fc4f30",
            2: "#FF1493",
            3: "#006400",
            4: "#FFD700",
            5: "#B8860B",
            6: "#008B8B",
            7: "#FF7F50",
            8: "#B22222",
            9: "#FF00FF",
            10: "#4169E1"
            }

    for nb_cluster in nb_clusters:

        # Plot the data and cluster silhouette comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)

        # The k-means plot
        kmeans = KMeans(n_clusters=nb_cluster)
        # Fit the algorithms to the features
        kmeans.fit(X_scale)
        # Compute the silhouette scores for each algorithm
        kmeans_silhouette = silhouette_score(X_scale, kmeans.labels_).round(2)
        km_colors = [fte_colors.get(label,"#696969") for label in kmeans.labels_]
        ax1.scatter(X_scale[features_column_name[0]], X_scale[features_column_name[1]], c=km_colors)
        ax1.set_title(f"{nb_cluster} clusters k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12})
        ax1.set_xlabel(features_column_name[0])
        ax1.set_ylabel(features_column_name[1])

        # The dbscan plot
        try:
            # Instantiate dbscan algorithms
            dbscan = DBSCAN(eps=(nb_cluster/10))
            # Fit the algorithms to the features
            dbscan.fit(X_scale)
            dbscan_silhouette = silhouette_score(X_scale, dbscan.labels_).round(2)
            db_colors = [fte_colors.get(label,"#696969") for label in dbscan.labels_]
            ax2.scatter(X_scale[features_column_name[0]], X_scale[features_column_name[1]], c=db_colors)
            ax2.set_title(f"{nb_cluster} clusters DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12})
            ax2.set_xlabel(features_column_name[0])
            ax2.set_ylabel(features_column_name[1])
        except Exception as error:
            if verbose:
                print(f"ERROR : {nb_cluster} clusters : dbscan_silhouette = {error}")
        if verbose:
            try:
                print(f"{nb_cluster} clusters : kmeans_silhouette = {kmeans_silhouette} <> dbscan_silhouette = {dbscan_silhouette}")
            except:
                pass
        fig.set_size_inches(18, 7, forward=True)
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