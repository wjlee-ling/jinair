import asyncio
from typing import AsyncIterator, Iterator, List, Optional
from playwright.async_api import async_playwright
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utils.user_agent import get_user_agent


class CustomAsyncChromiumLoader(BaseLoader):
    """Scrape HTML pages from URLs using a
    headless instance of the Chromium."""

    def __init__(
        self,
        urls: List[str],
        *,
        headless: bool = True,
        user_agent: Optional[str] = None,
        depth: int = 1,
    ):
        """Initialize the loader with a list of URL paths.

        Args:
            urls: A list of URLs to scrape content from.
            headless: Whether to run browser in headless mode.
            user_agent: The user agent to use for the browser

        Raises:
            ImportError: If the required 'playwright' package is not installed.
        """
        self.urls = urls
        self.headless = headless
        self.user_agent = user_agent or get_user_agent()
        self.depth = depth

        try:
            import playwright  # noqa: F401
        except ImportError:
            raise ImportError(
                "playwright is required for AsyncChromiumLoader. "
                "Please install it with `pip install playwright`."
            )

    async def _get_links_at_depth(self, url, current_depth=1):
        print("🍎" * current_depth)
        if current_depth > self.depth:
            return []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                page = await browser.new_page()
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
            except Exception as e:
                print(f"Failed to access {url}: {e}")
                return []

            if current_depth == self.depth:
                links = []
                link_elements = await page.query_selector_all("a")
                for element in link_elements:
                    try:
                        await element.hover()  # Hover over the element to reveal the URL if needed
                        link = await element.get_attribute("href")
                        if link:
                            links.append(link)
                    except Exception as e:
                        print(f"Failed to get link from element: {e}")
                await page.close()
                return links

            links = []
            link_elements = await page.query_selector_all("a")
            for element in link_elements:
                try:
                    await element.hover()  # Hover over the element to reveal the URL if needed
                    link = await element.get_attribute("href")
                    if link:
                        links.append(link)
                except Exception as e:
                    print(f"Failed to get link from element: {e}")

            unique_links = list(set(links))  # Remove duplicates

            all_links = []
            for link in unique_links:
                try:
                    sub_links = await self._get_links_at_depth(link, current_depth + 1)
                    all_links.extend(sub_links)
                except Exception as e:
                    print(f"Failed to access {link}: {e}")
                    continue

            await page.close()
            return all_links

    async def ascrape_playwright(self, url: str) -> str:
        """
        Asynchronously scrape the content of a given URL using Playwright's async API.

        Args:
            url (str): The URL to scrape.

        Returns:
            str: The scraped HTML content or an error message if an exception occurs.

        """
        results = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                page = await browser.new_page(user_agent=self.user_agent)
                await page.goto(url)

                ## Question-part
                header = await page.query_selector("header.article-header h1")
                if header:
                    header = await header.inner_text()
                else:
                    header = ""

                ## Answer-part
                answer = await page.query_selector("div.article-body")
                if answer:
                    answer = await answer.inner_text()
                else:
                    answer = ""

                ## combine Q & A
                results = f"Q: {header}\nA: {answer}"

            except Exception as e:
                results = f"Error: {e}"
            await browser.close()
        return results

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load text content from the provided URLs.

        This method yields Documents one at a time as they're scraped,
        instead of waiting to scrape all URLs before returning.

        Yields:
            Document: The scraped content encapsulated within a Document object.

        """
        for url in self.urls:
            children_links = asyncio.run(self._get_links_at_depth(url))
            print("🩷", children_links)
            for link in children_links:
                content = asyncio.run(self.ascrape_playwright(link))
                metadata = {"source": link}
                yield Document(page_content=content, metadata=metadata)

            # html_content = asyncio.run(self.ascrape_playwright(url))
            # metadata = {"source": url}
            # yield Document(page_content=html_content, metadata=metadata)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """
        Asynchronously load text content from the provided URLs.

        This method leverages asyncio to initiate the scraping of all provided URLs
        simultaneously. It improves performance by utilizing concurrent asynchronous
        requests. Each Document is yielded as soon as its content is available,
        encapsulating the scraped content.

        Yields:
            Document: A Document object containing the scraped content, along with its
            source URL as metadata.
        """
        # tasks = []
        for url in self.urls:
            children_links = self._get_links_at_depth(url)
            for link in children_links:
                content = self.ascrape_playwright(link)
                metadata = {"source": link}

                yield Document(page_content=content, metadata=metadata)

        # results = await asyncio.gather(*tasks)
        # for url, content in zip(self.urls, results):
        #     metadata = {"source": url}
        #     yield Document(page_content=content, metadata=metadata)


loader = CustomAsyncChromiumLoader(
    [
        "https://help.jinair.com/hc/ko/sections/4408903083673-%EC%98%88%EC%95%BD-%EA%B2%B0%EC%A0%9C"
        # "https://help.jinair.com/hc/ko/categories/4408759363353-%EC%9E%90%EC%A3%BC-%EB%AC%BB%EB%8A%94-%EC%A7%88%EB%AC%B8-FAQ"
    ],
    depth=1,
)
docs = loader.load()
for doc in docs:
    print(doc.metadata)
