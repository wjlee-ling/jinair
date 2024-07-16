import os
from operator import itemgetter
from typing import Union, List

from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from urllib.parse import urljoin


load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "jinair-test"

BASE_URL = "https://help.jinair.com/"
_TEMPLATE = """\
As a helpful assistant, your task is to provide an answer to the question of the user based on the given query and HTML content. \
The HTML content may or may not contain all the relevant information enough to answer the question. \
Only when you can find the full information you need to answer the question in the current HTML content, you should provide the **final answer** given the content right away. \ 
Or else, you should provide the user with the (up to) 2 href links with inner texts that seem most relevant to the question. You should return the href links in a Python list. 

Make sure to return the answer of `str` or href links of `list[str]` in the output format without any prefix, suffix and/or explanations to your answer.
Make sure to extract the href links from the HTML content when returning them. DO NOT make up any href links.
Make sure to extract and return href links when there is relevant context for the question and an answer but the context trails off or is not finished.
Make sure to give a final answer without any tags to the question if you can answer the question based on the html. The answer should be in accordance with the question asked.
Make sure not to return the href links in the Markdown Link syntax. Just return the href links as a python list of strings when you return href links instead of a string final answer.
Make sure to return something like "죄송합니다. 질문에 대한 문서를 찾을 수 없습니다." only when you cannot find **any** relevant information to the question and **any href links** in the HTML content at all."""


def _join_urls(response):
    import re

    # Regular expression to find markdown links
    pattern = r"\[([^\]]+)\]\(\1\)"
    links = re.findall(pattern, response)

    try:
        if links:
            return [urljoin(BASE_URL, link) for link in links]
        elif isinstance(eval(response), list):
            urls = list(set(eval(response)))
            return [urljoin(BASE_URL, url) for url in urls]
        else:
            return eval(response)
    except:
        return response


def get_web_scraper(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _TEMPLATE),
            ("user", "html content:\n{html_content}\nQuestion:\n{query}"),
        ]
    )

    web_scraper_chain = (
        {
            "query": itemgetter("query"),
            "html_content": itemgetter("html_content"),
        }
        | prompt
        | llm
        | StrOutputParser()
        | _join_urls
    ).with_config(run_name="web_scraper")

    return web_scraper_chain


def _extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser").body

    for data in soup(["style", "script", "nav", "footer", "svg"]):
        # Remove style, script tags
        data.decompose()

    def has_direct_text(tag):
        return any(isinstance(child, str) and child.strip() for child in tag.children)

    filtered_tags = []
    for tag in soup.main.find_all(True):
        if tag.name == "a" and "href" in tag.attrs:
            filtered_tags.append(f'<a href="{tag["href"]}">{tag.get_text()}</a>')
        elif has_direct_text(tag):
            filtered_tags.append(tag)

    result_html = [str(tag) for tag in filtered_tags]

    # tags_with_inner_text = [tag for tag in soup.main.find_all(True) if tag.text.strip()]
    # a_tags_with_href = [
    #     f'<a href="{a_tag["href"]}">{a_tag.get_text()}</a>'
    #     for a_tag in soup.main.find_all("a", href=True)
    # ]
    # result_tags = tags_with_inner_text + a_tags_with_href
    # result_tags = list(set(result_tags))
    return "\n".join(result_html)


def run_web_scraping(query, llm, root_url: Union[str, List]):
    web_scraper_chain = get_web_scraper(llm)
    if isinstance(root_url, str):
        loader = AsyncChromiumLoader([root_url])
        html_content = loader.load()[0].page_content
        html = _extract_body_content(html_content)
    else:
        loader = AsyncChromiumLoader(root_url)
        html_contents = [
            _extract_body_content(page.page_content) for page in loader.load()
        ]
        html = "\n\n".join(html_contents)

    resp = web_scraper_chain.invoke(
        {
            "query": query,
            "html_content": str(html),
        }
    )

    if isinstance(resp, str):
        return resp
    elif isinstance(resp, list):
        return run_web_scraping(query, llm, resp)


# llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# web_scraper_chain = get_web_scraper(llm)

# resp = run_web_scraping(
#     llm,
#     # "https://help.jinair.com/hc/ko/articles/23199933265689-%ED%8E%B8%EC%9D%98%EC%A0%90-%EA%B2%B0%EC%A0%9C%EB%A1%9C-%EC%98%88%EC%95%BD%EC%9D%84-%ED%96%88%EB%8A%94%EB%8D%B0-%EC%98%88%EC%95%BD%EC%99%84%EB%A3%8C-%EC%9D%B4%EB%A9%94%EC%9D%BC%EC%9D%84-%EB%AA%BB-%EB%B0%9B%EC%95%98%EC%96%B4%EC%9A%94",
#     "https://help.jinair.com/hc/ko/categories/4408759363353-%EC%9E%90%EC%A3%BC-%EB%AC%BB%EB%8A%94-%EC%A7%88%EB%AC%B8-FAQ",
# )
# print(resp)


# queries = [
#     "제주도 사는 사람인데 받을 수 있는 혜택은?",
#     # "비행기표 변경 방법",
#     # "셀프 체크인 어떻게 해요?",
#     # "기내식 신청하려고 하는데",
# ]

# for query in queries:
#     response = web_scraper_chain.invoke({"query": query, "html_content": html})
#     print(response)
