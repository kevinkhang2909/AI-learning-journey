import httpx
from bs4 import BeautifulSoup
import polars as pl


def get_news(url: str) -> list:
    limit_news = 50
    # session = requests.Session()
    response = httpx.get(url)
    print('Done getting news')
    soup = BeautifulSoup(response.content, 'html.parser')
    article = soup.select("article.item-news", limit=limit_news)

    lst = []
    for element in article:
        title = element.select("h3.title-news > a")
        description = element.select("p.description > a")
        for x in range(len(title)):
            dict_ = {
                'title': title[x]['title'],
                'link': title[x]['href'],
                'description': description[x].text
            }
            lst.append(dict_)
    return lst


# url = 'https://vnexpress.net/'
# url = 'https://cafef.vn/'
# lst = get_news(url)
