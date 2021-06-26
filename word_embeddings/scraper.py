import json
import traceback
import pandas as pd

from functools import wraps
from bs4 import BeautifulSoup
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from scraper_api import ScraperAPIClient


def try_except(func):
    """
    A decorator for exception handing

    :param func: function for which exception handing will be performed
    :type func: function
    :return: wrapped function
    :rtype: function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            value = func(*args, **kwargs)
            return value
        except Exception:
            traceback.print_exc()
            return None

    return wrapper


def get_article_metadata(client, query, start_year=2015, filepath="data/temp.csv"):
    """ 
    Fetch article metadata matching a search query from scopus
    :param client: represents a client interface to api.elsevier.com
    :type client: ElsClient
    :param query: search query e.g. "TITLE-ABS-KEY ( micromorphic AND homogenization )"
    :type query: str
    :param start_year: start year to search articles, defaults to 2015
    :type start_year: int, optional
    :param filepath: path to the file where results will be stored, defaults to 'data/temp.json'
    :type filepath: str, optional

    :return dataframe with retrieved article details
    """
    query += "AND PUBYEAR > {}".format(str(start_year))
    doc_search = ElsSearch(query, "scopus")
    doc_search.execute(client, get_all=False)
    if len(doc_search.results) <= 1:
        print("No result found")
        return pd.DataFrame()
    print("# of article retrieved: {}".format(len(doc_search.results)))

    # save doc search results
    # with open(filepath, "w") as f:
    #    json.dump(doc_search.results, f)

    # data cleaning

    article_dict = dict()
    for idx, article in enumerate(doc_search.results):
        tmp_dict = dict()
        tmp_dict["scopus_id"] = int(article["prism:url"].split("/")[-1])
        tmp_dict["title"] = article["dc:title"]
        tmp_dict["first_author"] = article["dc:creator"]
        tmp_dict["affiliation"] = [item["affilname"] for item in article["affiliation"]]
        tmp_dict["publication_name"] = article["prism:publicationName"]
        tmp_dict["publication_date"] = article["prism:coverDate"]
        tmp_dict["citation_count"] = article["citedby-count"]
        for link in article["link"]:
            if link["@ref"] == "scopus":
                tmp_dict["abstract_link"] = link["@href"]
            elif link["@ref"] == "scopus-citedby":
                tmp_dict["cited_by_link"] = link["@href"]
        article_dict[idx] = tmp_dict
    df = pd.DataFrame(article_dict).T
    df.to_csv(filepath, index=False)
    return df


class ArticleDataExtractor:
    """
    A class to facilitate extracion of scientific publication data from scopus
    """

    def __init__(self, client):
        """
        initialization for the class

        :param client: a ScraperAPIClient
        :type client: ScraperAPIClient
        """
        self.client = client
        self.soup = BeautifulSoup()

    def fetch_data(self, url):
        """
        fetch data from the web using scraper api

        :param url: url from which data will be fetched
        :type url: str
        :return: response code
        :rtype: int
        """
        result = self.client.get(url=url, render=False, autoparse=True)
        self.soup = BeautifulSoup(result.text)
        return result.status_code

    @try_except
    def _get_title(self):
        """
        Get article title

        :return: article title
        :rtype: str
        """
        return self.soup.title.text.split("Document details -")[-1].strip()

    @try_except
    def _get_abstract(self):
        """
        Get article abstract

        :return: abstract
        :rtype: str
        """
        tmp = self.soup.find("section", id="abstractSection")
        abstract = "".join([para.text.strip() for para in tmp.findAll("p")])
        return abstract

    @try_except
    def _get_authors(self):
        """
        Get article authors

        :return: a list of authors
        :rtype: List
        """
        tmp = self.soup.find("section", id="authorlist")
        authors = [
            auth.text.strip() for auth in tmp.findAll("span", class_="previewTxt")
        ]
        return authors

    @try_except
    def _get_author_keywords(self):
        """
        Get article keywords as specified by authors

        :return: keyword list
        :rtype: List
        """
        tmp = self.soup.find("section", id="authorKeywords")
        author_kws = [kw.text.strip() for kw in tmp.findAll("span")]
        return author_kws

    @try_except
    def _get_index_keywords(self):
        """
        Get article keywords as specified by scopus

        :return: keyword list
        :rtype: List
        """
        tmp = self.soup.find("section", id="indexedKeywords")
        index_kws = [kw.text.strip() for kw in tmp.findAll("span", class_="badges")]
        return index_kws

    @try_except
    def _get_citedby_articles(self):
        """
        Get a sample of recent articles which cites the current article

        :return: info on citedby articles
        :rtype: dict
        """
        tmp = self.soup.find("div", id="recordPageBoxes")
        citedby_dict = dict()
        for idx, kw in enumerate(tmp.findAll("div", class_="recordPageBoxItem")):
            authors = []
            for auth in kw.findAll(
                "span",
                title="Search for all documents by this author: Subscription required",
            ):
                authors.append(auth.text.strip())
            title = kw.find("span", class_="authorLinks").text.strip()
            journal = kw.find("span", class_="italicText").text.strip()
            citedby_dict[idx] = {"authors": authors, "title": title, "journal": journal}
        return citedby_dict

    def execute(self, url):
        """
        orchestrator method for extracting the article information 

        :param url: url from which data will be extracted
        :type url: str
        :return: a dict with article information
        :rtype: dict
        """
        status_code = self.fetch_data(url)
        if status_code != 200:
            print("Error in fetching data from Scopus! Exiting")
            return dict()

        article = dict()
        article["title"] = self._get_title()
        article["authors"] = self._get_authors()
        article["abstract"] = self._get_abstract()
        article["author_kws"] = self._get_author_keywords()
        article["index_kws"] = self._get_index_keywords()
        article["cited_by"] = self._get_citedby_articles()

        return article


if __name__ == "__main__":
    ## Load configuration
    with open("config.json") as f:
        config = json.load(f)

    ## Initialize client
    this_client = ElsClient(config["apikey"])

    search_query = "TITLE-ABS-KEY ( FE  AND NN )"
    get_article_metadata(this_client, search_query)
