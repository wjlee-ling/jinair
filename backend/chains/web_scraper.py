from operator import itemgetter

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())


_TEMPLATE = """\
As a helpful assistant, your task is to provide the user with the information they need based on the given query and HTML content. \
The HTML content may or may not contain all the relevant information to the query or just href links to the web pages where the information is available. \
Only when you are fully given the information you need to answer the query in the current HTML content, you should provide the answer right away. \ 
Or else, you should provide the user with the href links to the web pages where the information is available.

Make sure to return the answer or href links in the output format without any prefix, suffix and/or explanations to your answer.
Make sure to extract the href links from the HTML content when returing them. DO NOT make up any href links.
Make sure to return all the href links in a Python list format if you are uncertain which link is the most relevant to the query."""


def get_web_scraper(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _TEMPLATE),
            ("user", "html content:\n{html_content}\n\nquery:\n{query}"),
        ]
    )

    web_scraper_chain = (
        {
            "query": itemgetter("query"),
            "html_content": itemgetter("html_content"),
        }
        | prompt
        | llm
    ).with_config(run_name="web_scraper")

    return web_scraper_chain


llm = ChatOpenAI(model="gpt-4o")
web_scraper_chain = get_web_scraper(llm)

html = """<main role="main">
    <div class="container-divider"></div>
<div class="container">
  <nav class="sub-nav">    
    	<ol class="breadcrumbs">
  
    <li title="고객서비스">
      
        <a href="/hc/ko">고객서비스</a>
      
    </li>
  
    <li title="자주 묻는 질문 (FAQ)">
      
        자주 묻는 질문 (FAQ)
      
    </li>
  
</ol>

    <div class="search-container">
      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" focusable="false" viewBox="0 0 12 12" class="search-icon">
        <circle cx="4.5" cy="4.5" r="4" fill="none" stroke="currentColor"></circle>
        <path stroke="currentColor" stroke-linecap="round" d="M11 11L7.5 7.5"></path>
      </svg>
      <form role="search" class="search" data-search="" action="/hc/ko/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="✓" autocomplete="off"><input type="search" name="query" id="query" placeholder="검색" aria-label="검색" class="placeholder"></form>
    </div>
  </nav>

  <div class="category-container">
    <div class="category-content">
      <header class="page-header">
        <h1>자주 묻는 질문 (FAQ)</h1>
        
          <p class="page-header-description">고객님께서 자주 문의 하시는 질문 및 답변을 모아 놓았습니다.</p>
        
      </header>

      <div id="main-content" class="section-tree">
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408767164953-%ED%95%AD%EA%B3%B5%EA%B6%8C-%EC%98%88%EB%A7%A4">항공권 예매</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4408903083673-%EC%98%88%EC%95%BD-%EA%B2%B0%EC%A0%9C" class="article-list-link">예약/결제</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4408903098265-%EB%B3%80%EA%B2%BD-%ED%99%98%EB%B6%88" class="article-list-link">변경/환불</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409095439769-%EC%82%AC%EC%A0%84%EC%A2%8C%EC%84%9D%EC%A7%80%EC%A0%95" class="article-list-link">사전좌석지정</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409090914201-%EC%A7%80%EB%8B%88%ED%94%8C%EB%9F%AC%EC%8A%A4-JINI-PLUS-%EC%A2%8C%EC%84%9D" class="article-list-link">지니플러스(JINI PLUS) 좌석</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103310489-%EB%AC%B6%EC%9D%8C-%ED%95%A0%EC%9D%B8" class="article-list-link">묶음 할인</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409090925081-%EA%B8%B0%ED%83%80-%EC%98%81%EC%88%98%EC%A6%9D-%EB%B0%9C%EA%B8%89-%EB%93%B1" class="article-list-link">기타(영수증 발급 등)</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408826770201-%ED%95%A0%EC%9D%B8%EC%A0%9C%EB%8F%84">할인제도</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409090966425-%EA%B0%80%EC%A1%B1%EC%9A%B4%EC%9E%84-%ED%95%A0%EC%9D%B8%EC%A0%9C%EB%8F%84" class="article-list-link">가족운임 할인제도</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103330713-%EC%A0%9C%EC%A3%BC-%EC%9E%AC%EC%99%B8-%EB%AA%85%EC%98%88%EB%8F%84%EB%AF%BC-%ED%95%A0%EC%9D%B8%EC%A0%9C%EB%8F%84" class="article-list-link">제주·재외·명예도민 할인제도</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409090944281-%EA%B8%B0%ED%83%80-%EC%A0%9C%ED%9C%B4-%ED%95%A0%EC%9D%B8" class="article-list-link">기타 제휴 할인</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408826780185-%ED%94%84%EB%A1%9C%EB%AA%A8%EC%85%98">프로모션</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409095476505-%EC%A7%84MARKET" class="article-list-link">진MARKET</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408896537369-%EC%B2%B4%ED%81%AC%EC%9D%B8-%EC%88%98%EC%86%8D">체크인(수속)</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103347609-%EA%B3%B5%ED%95%AD-%EC%B2%B4%ED%81%AC%EC%9D%B8" class="article-list-link">공항 체크인</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103345945-%EC%9B%B9-%EB%AA%A8%EB%B0%94%EC%9D%BC-%EC%B2%B4%ED%81%AC%EC%9D%B8" class="article-list-link">웹·모바일 체크인</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103344153-%EC%85%80%ED%94%84-%EC%B2%B4%ED%81%AC%EC%9D%B8" class="article-list-link">셀프 체크인</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409090978457-%EC%97%AC%ED%96%89%EC%84%9C%EB%A5%98" class="article-list-link">여행서류</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408896539033-%EC%88%98%ED%95%98%EB%AC%BC">수하물</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091017753-%ED%9C%B4%EB%8C%80-%EC%88%98%ED%95%98%EB%AC%BC" class="article-list-link">휴대 수하물</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409095490329-%EC%9C%84%ED%83%81-%EC%88%98%ED%95%98%EB%AC%BC" class="article-list-link">위탁 수하물</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103353241-%EA%B8%B0%EB%82%B4-%EC%9C%A0%EC%8B%A4%EB%AC%BC" class="article-list-link">기내 유실물</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091003161-%EC%88%98%ED%95%98%EB%AC%BC-%EB%B0%B0%EC%83%81" class="article-list-link">수하물 배상</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408910174873-%EB%8F%84%EC%9B%80%EC%9D%B4-%ED%95%84%EC%9A%94%ED%95%98%EC%8B%A0-%EA%B3%A0%EA%B0%9D">도움이 필요하신 고객</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091057817-%EC%9C%A0%EC%95%84%EB%8F%99%EB%B0%98-%EC%9E%84%EC%82%B0%EB%B6%80-%EA%B3%A0%EA%B0%9D" class="article-list-link">유아동반·임산부 고객</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409095501337-%EB%B0%98%EB%A0%A4%EB%8F%99%EB%AC%BC%EA%B3%BC-%EC%97%AC%ED%96%89%ED%95%98%EB%8A%94-%EA%B3%A0%EA%B0%9D" class="article-list-link">반려동물과 여행하는 고객</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091030553-%EB%AA%B8%EC%9D%B4-%EB%B6%88%ED%8E%B8%ED%95%98%EC%8B%A0-%EA%B3%A0%EA%B0%9D" class="article-list-link">몸이 불편하신 고객</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409095495065-%EB%B3%B4%EC%A1%B0%ED%98%B8%ED%9D%A1%EC%9E%A5%EC%B9%98-%EC%82%AC%EC%9A%A9-%EA%B3%A0%EA%B0%9D" class="article-list-link">보조호흡장치 사용 고객</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408910177305-%EA%B8%B0%EB%82%B4%EC%84%9C%EB%B9%84%EC%8A%A4">기내서비스</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091071257-%EC%A7%80%EB%8B%88-%EA%B8%B0%EB%82%B4%EC%8B%9D" class="article-list-link">지니 기내식</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409095508761-%EC%A7%80%EB%8B%88%EC%8A%A4%ED%86%A0%EC%96%B4" class="article-list-link">지니스토어</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409095506969-%EC%A7%80%EB%8B%88-DUTY-FREE" class="article-list-link">지니 DUTY FREE</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091059865-%EA%B8%B0%ED%83%80-%EC%84%9C%EB%B9%84%EC%8A%A4" class="article-list-link">기타 서비스</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408904777881-%ED%99%88%ED%8E%98%EC%9D%B4%EC%A7%80">홈페이지</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091082777-%ED%9A%8C%EC%9B%90%EA%B0%80%EC%9E%85-%EB%B3%80%EA%B2%BD" class="article-list-link">회원가입·변경</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103371673-%EB%B9%84%ED%9A%8C%EC%9B%90" class="article-list-link">비회원</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103370137-%ED%83%88%ED%87%B4" class="article-list-link">탈퇴</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091073561-%EA%B8%B0%ED%83%80" class="article-list-link">기타</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408904779929-%EA%B3%B5%EB%8F%99%EC%9A%B4%ED%95%AD">공동운항</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103373593-%EB%8C%80%ED%95%9C%ED%95%AD%EA%B3%B5-%EA%B3%B5%EB%8F%99%EC%9A%B4%ED%95%AD" class="article-list-link">대한항공 공동운항</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408904783769-%EB%82%98%EB%B9%84%ED%8F%AC%EC%9D%B8%ED%8A%B8">나비포인트</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091091609-%EC%A0%81%EB%A6%BD-%EC%86%8C%EB%A9%B8" class="article-list-link">적립·소멸</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409103383833-%EC%82%AC%EC%9A%A9-%EB%B3%B4%EB%84%88%EC%8A%A4-%ED%95%AD%EA%B3%B5%EA%B6%8C" class="article-list-link">사용(보너스 항공권)</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408910187033-%EA%B8%B0%ED%83%80">기타</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091097241-%ED%9A%8C%EC%82%AC%EC%95%88%EB%82%B4" class="article-list-link">회사안내</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091094809-%EA%B3%A0%EA%B0%9D%EC%84%BC%ED%84%B0" class="article-list-link">고객센터</a>
                  </li>
                
                  <li class="article-list-item article-promoted">                    
                    <a href="/hc/ko/sections/4409091093273-%EC%A0%9C%ED%9C%B4-%EB%AC%B8%EC%9D%98-%EB%93%B1" class="article-list-link">제휴·문의 등</a>
                  </li>
                
              </ul>
          </section>
        
          <section class="section">
            <h2 class="section-tree-title">
              <a href="/hc/ko/sections/4408904791577-%EC%84%9C%EB%A5%98%EC%96%91%EC%8B%9D%ED%95%A8">서류양식함</a>
              <a href="#" class="btn_cat">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" focusable="false" viewBox="0 0 16 16" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M5 14.5l6.1-6.1c.2-.2.2-.5 0-.7L5 1.5"></path>
              </svg>
              </a>
            </h2>            
              <ul class="article-list">
                
              </ul>
          </section>
        
      </div>
    </div>
  </div>  
</div>
<script type="text/javascript">
  
  if (window.innerWidth < 768) {
    const section = document.querySelectorAll('.sections');
    const btn = document.querySelectorAll('.section .btn_cat');                             
    let clickCk = 0;
    console.log('mobile');
                              
   for (var i = 0; i < section.length; i++) {
    	if (0 >= section[i].querySelectorAll('ul li').length) {
  				section[i].querySelector('.btn_cat').style.display = 'none';
  		}                        
    }                             
                              
    for (var i = 0; i < btn.length; i++) {
      btn[i].addEventListener('click', function (e) {
        if (this.closest('.section').classList.contains('on')) {
          this.closest('.section').classList.remove('on');
          //clickCk = 0;
        } else {
          this.closest('.section').classList.add('on');
          //clickCk = 1;
        }
  			e.preventDefault();
      });
    }
  }
                                   
  $(document).ready(function() {  	
    $(".breadcrumbs").find("li[title ~= 'KMS']").parent().children().attr("style", 'display:none');
    $(".breadcrumbs").find("li[title ~= 'KMS']").parent().children().first().attr("style", 'display:inline');
  	$(".breadcrumbs").find("li[title ~= 'KMS']").parent().children().last().attr("style", 'display:inline');
  });                                  
</script>
  </main>"""

queries = [
    "제주도 사는 사람인데 받을 수 있는 혜택은?",
    "비행기표 변경 방법",
    "셀프 체크인 어떻게 해요?",
    "기내식 신청하려고 하는데",
]

for query in queries:
    response = web_scraper_chain.invoke({"query": query, "html_content": html})
    print(response)
