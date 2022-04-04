from urllib import request
import bs4
from selenium import webdriver
import time
from os import environ
from datetime import datetime

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              COMMON SCRAPPING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_page(url, verbose=0):
    """Retrieve the HTML page

    Args:
        url (str): the page URL
        verbose (int, optional): Log level. Defaults to 0.

    Returns:
        BeautifulSoup: the page
    """
    if verbose>1:
        print("Request url :", url)
    req = request.Request(url, headers = {'User-Agent' : 'Mozilla/5.0'})
    html = request.urlopen(req).read()
    return bs4.BeautifulSoup(html, "lxml")

def get_page_links(page, base_url="", class_=None, liste_urls=None, verbose=0):
    """Return all page or section links
    Args:
        page (BeautifulSoup): the page or the section of the page
        base_url (str, optional): the url base. Defaults to "".
        class_ (str, optional): The link class. Defaults to None.
        liste_urls (set(str), optional): The url list to complete. Defaults to None.
        verbose (int, optional): log level. Defaults to 0.

    Raises:
        AttributeError: if page is None

    Returns:
        set(str): link list
    """
    if page is None:
        raise AttributeError("page expected")
    
    if liste_urls is None:
        liste_urls =set()
    nb_url_start = len(liste_urls)
    liens_list = page.findAll('a')
    if class_ is None:
        liens_list = page.findAll('a')
    else:
        liens_list = page.findAll('a', {'class': class_})
    
    # Récupération de tous les liens de la page
    for lien in liens_list:
        link = lien.get('href')
        if base_url is not None and len(base_url) >0 and base_url not in link:
            link = base_url+link
        liste_urls.add(link)
    
    if verbose>1:
        print(len(liste_urls)-nb_url_start, " URLs found")
    return liste_urls

def get_selenium_firefox_driver(url, gecko_driver_path=None, verbose=0):
    """Create and return the firefox driver for selenium

    Args:
        url (str): URL to load
        gecko_driver_path (str, optional): the path to the gecko_driver. Defaults to None, so use the environnement variable : `GECKO_DRIVER_PATH`.
        verbose (int, optional): log level. Defaults to 0.

    Returns:
        WebDriver: firefox webdriver
    """
    if gecko_driver_path is None:
        if verbose:
            print('get_selenium_driver > No Gecko driver path, so use the environnement variable : `GECKO_DRIVER_PATH`')
        gecko_driver_path = environ.get('GECKO_DRIVER_PATH')
        if gecko_driver_path is None:
            raise Exception("No `GECKO_DRIVER_PATH` environment varibale define. This variable is mandatory to use selenium on firefox")
    driver = webdriver.Firefox(executable_path=gecko_driver_path)
    driver.get(url)
    time.sleep(5)
    return driver


def save_df_in_file(df_to_save, file_path):
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H_%M_%S")
    df_to_save.to_csv(file_path+"_" + date_time + '.csv', sep=',', index=False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    verbose = 1
    pass