from collections import defaultdict
import numpy as np
from scrapping_util import get_page, save_df_in_file
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from bonheur_bed_ara import complete_df_with_country_datas
import sys
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\")
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\ara_commons\\")
from countries.country_constants import get_country_data

def add_country_line(tr_balise, dic=None, verbose=0):
    res = None
    if dic is None:
        res = defaultdict(list)
    else:
        res = dic.copy()

    col_order = ["pays", "2015", "2016", "2017", "2018", "2019", "2020"]

    if tr_balise is not None:
        i = 0
        for child in tr_balise.findChildren():
            if "td" == child.name and child.get_text() is not None:
                value = child.get_text().strip()
                if len(value)>0:
                    if ("(201" in value and ")" == value[-1]) or ("-" in value and len(value)==1):
                        value = np.nan
                    key = col_order[i]
                    res[key].append(value)
                    i += 1
    # Ajout de la colonne country
    if len(res)>0:
        country_name = get_country_data(country_name=res["pays"], verbose=verbose)
        res["country"].append(country_name)
    return res


def get_historique_dic(url="https://atlasocio.com/classements/societe/bonheur/classement-etats-par-indice-de-bonheur-monde.php", verbose=0):
    """Retrieve historique datas

    Args:
        url (str): the article page url
        tags (str, optional): the tags. Defaults to None.
        journal (Journal, optional): The paper. Defaults to None.
        verbose (int, optional): log level. Defaults to 0.

    Raises:
        AttributeError: if url is missing

    Returns:
        Article: the article
    """
    if url is None or len(url)==0:
        raise AttributeError("URL expected")

    page = get_page(url)
    table = page.find('table', {'class': "responsive-table-all alternative rang"})
    
    res = defaultdict(list)
    if table is not None:
        try:
            table = table.find("tbody")
            if table is not None:
                for tr_balise in tqdm(table.findAll('tr')):
                    try:
                        res = add_country_line(tr_balise, dic=res, verbose=verbose)
                    except Exception as error:
                        print(error)
        except Exception as error:
            print(error)
    
    df = pd.DataFrame.from_dict(res)
    
    if verbose:
        print(df.head())
    save_df_in_file(df, file_path=r"C:\Users\User\WORK\workspace-ia\PROJETS\projet_bonheur_bed\dataset\evolution_score_altasocio")
    return df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from os import getcwd

if __name__ == "__main__":

    verbose = 1
    # df = get_historique_dic(verbose=verbose)
    file_path = getcwd() + "\\"
    data_set_path = 'C:\\Users\\User\\WORK\\workspace-ia\\PROJETS\\projet_bonheur_bed\\dataset\\'
    print(f"Current execution path : {file_path}")
    print(f"Dataset path : {data_set_path}")

    data_evolution_name = "evolution_score_altasocio_2022-03-31.csv"
    df_evolution_orgin = pd.read_csv(data_set_path+data_evolution_name, sep=',')
    try:
        df_evolution_orgin= df_evolution_orgin.drop("country_official", axis=1)
    except:
        pass
    df_evolution_orgin = complete_df_with_country_datas(df_evolution_orgin, verbose=verbose)
    save_df_in_file(df_evolution_orgin, file_path=r"C:\Users\User\WORK\workspace-ia\PROJETS\projet_bonheur_bed\dataset\evolution_score_altasocio")
    print("END")

            