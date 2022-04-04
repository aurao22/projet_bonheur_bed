from collections import defaultdict
import numpy as np
from scrapping_util import get_page, save_df_in_file
from tqdm import tqdm
from collections import defaultdict
import pandas as pd


indices_url_dic = {
    "bourse":"https://fr.countryeconomy.com/marches/bourse",
    "chomage":"https://fr.countryeconomy.com/marche-du-travail/chomage",
    "dette_public":"https://fr.countryeconomy.com/gouvernement/dette",
    "deficit_public":"https://fr.countryeconomy.com/gouvernement/deficit",
    "Enquete_Population_Active":"https://fr.countryeconomy.com/marche-du-travail/enquete-population-active",
    "Consumer_Price_Index":"https://fr.countryeconomy.com/node/311",
    "Matieres_Premiere":"https://fr.countryeconomy.com/marches/matieres-premieres",
    "Taux_Change":"https://fr.countryeconomy.com/marches/monnaies",
    "Obligations":"https://fr.countryeconomy.com/marches/obligations",
    "PIB":"https://fr.countryeconomy.com/gouvernement/pib",
    "SMIC":"https://fr.countryeconomy.com/marche-du-travail/salaire-minimum-national",
    "Taux_Interet":"https://fr.countryeconomy.com/marche-monetaire/taux-interet"}

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
    return res


def get_indices_df(indices_url_dic, verbose=0):
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
    if indices_url_dic is None:
        raise AttributeError("indices_url_dic expected")

    country_dic = defaultdict(defaultdict(list))
    for tag, url in indices_url_dic.items():
        page = get_page(url)
        table = page.find('table', {'class': "responsive-table-all alternative rang"})
        
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
    
    res = defaultdict(list)
    for country, dic in country_dic.items():
        res["pays"].append(country)
        for indice, val in dic.items():
            res[indice].append(val)

    df = pd.DataFrame.from_dict(res)
    
    if verbose:
        print(df.head())
    save_df_in_file(df, file_path=r"C:\Users\User\WORK\workspace-ia\PROJETS\projet_bonheur_bed\dataset\indice_2021")
    return df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":

    verbose = 1
    df = get_historique_dic(verbose=verbose)
    print("END")

            

