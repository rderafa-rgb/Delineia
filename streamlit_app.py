# -*- coding: utf-8 -*-

import streamlit as st
from streamlit import fragment
import base64

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Delinéia",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS CUSTOMIZADO (BOTÕES VERDES) ====================
st.markdown("""
<style>
    /* Força a barra de rolagem a estar sempre presente, evitando pulos laterais */
    html {
        overflow-y: scroll;
    }
            
    /* Centralizar texto de expanders */
    .streamlit-expanderHeader {
        justify-content: center;
        text-align: center;
        font-weight: bold;
    }
    
    /* CORREÇÃO: Forçar quebra de palavras longas no sidebar */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stExpander,
    [data-testid="stSidebar"] a {
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    /* Botões primários em verde claro */
    .stButton > button[kind="primary"] {
        background-color: #10b981 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #059669 !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:active {
        background-color: #047857 !important;
    }
    
    /* Form submit buttons */
    .stFormSubmitButton > button {
        background-color: #10b981 !important;
        color: white !important;
        border: none !important;
    }
    
    .stFormSubmitButton > button:hover {
        background-color: #059669 !important;
    }
    
    /* Download buttons com type="primary" */
    .stDownloadButton > button[kind="primary"] {
        background-color: #10b981 !important;
        color: white !important;
    }  
</style>
""", unsafe_allow_html=True)

# ======================== OUTROS IMPORTS ========================
from datetime import datetime, timezone, timedelta 
from research_pipeline import ResearchScopePipeline, OpenAlexClient, CooccurrenceAnalyzer, OPENALEX_EMAIL
from pdf_generator import generate_pdf_report
import pandas as pd
import networkx as nx
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import json
import zipfile
from io import BytesIO
import numpy as np
from scipy import stats
import gspread
from google.oauth2.service_account import Credentials
import uuid
import time as time_module
import matplotlib.pyplot as plt
import export_utils as exp
from export_utils import generate_excel, generate_bibtex, generate_ris, generate_pajek_net
import streamlit.components.v1 as components
import os
# Adiciona manualmente o caminho do Graphviz ao Python (Ajuste se instalou em outro lugar)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
import io
import tempfile
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
import gc

# ==================== FUNÇÕES AUXILIARES GLOBAIS ====================

def extract_concept_metadata(articles: list) -> dict:
    """
    Extrai metadados ricos (Score e Level) dos artigos brutos do OpenAlex.
    Essencial para o Histórico Rico e o Tesauro Visual.
    """
    from collections import defaultdict
    concept_data = defaultdict(lambda: {'scores': [], 'levels': [], 'count': 0})
    
    for article in articles:
        # Proteção contra artigos sem conceitos
        for concept in article.get('concepts', []):
            name = concept.get('display_name', '')
            if name:
                try:
                    score = float(concept.get('score', 0))
                    level = float(concept.get('level', 0))
                    
                    concept_data[name]['scores'].append(score)
                    concept_data[name]['levels'].append(level)
                    concept_data[name]['count'] += 1
                except:
                    continue

    metadata = {}
    for name, data in concept_data.items():
        # Calcula médias
        avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
        avg_level = sum(data['levels']) / len(data['levels']) if data['levels'] else 0
        
        metadata[name] = {
            'freq': data['count'],
            'score': avg_score,
            'level': avg_level
        }
    return metadata

# ========================= BASE64 =============================

def get_base64_image(image_path):
    """Converte imagem local para string base64 para uso em HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

def limpar_memoria():
    """Força coleta de lixo"""
    gc.collect()

# ==================== FUNÇÕES COM CACHE (OTIMIZAÇÃO DE MEMÓRIA) ====================

@st.cache_resource
def get_pipeline_instance():
    """Cache da instância do pipeline para não recriar objetos pesados."""
    return ResearchScopePipeline(OPENALEX_EMAIL)

def run_cached_pipeline(nome, tema, questao, kws, genero, google_academico=""):
    pipe = ResearchScopePipeline(OPENALEX_EMAIL)
    # A função process retorna dicionários e grafos NetworkX, que o Streamlit serializa bem
    return pipe.process(nome, tema, questao, kws, genero=genero, google_academico=google_academico)

@st.cache_data(ttl="1h", show_spinner=False)
def generate_cached_pdf(form_data, result, selected_concepts, suggested_keywords, suggested_strings, badges):
    """Cache da geração do PDF para evitar recriação do binário."""
    return generate_pdf_report(
        form_data=form_data,
        result=result,
        selected_concepts=selected_concepts,
        suggested_keywords=suggested_keywords,
        suggested_strings=suggested_strings,
        badges=badges
    )

def run_cached_thematic_map(graph_data, concepts_lists, method, min_size):
    """
    Executa a análise de mapa temático e retorna os dados prontos.
    Isso evita reprocessar a clusterização se os parâmetros não mudarem.
    """
    from thematic_map_module import ThematicMapAnalyzer
    
    # Cria o analisador e detecta clusters
    tm_analyzer = ThematicMapAnalyzer(graph_data, concepts_lists)
    tm_analyzer.detect_clusters(method=method, min_size=min_size)
    
    # Retorna o DataFrame de métricas E a lista de clusters (necessária para o gráfico)
    return tm_analyzer.analyze_clusters(), tm_analyzer.clusters

@st.cache_resource
def get_openalex_client():
    return OpenAlexClient(OPENALEX_EMAIL)

@st.cache_data(ttl="1h")
def search_openalex_cached(query, limit, min_score, min_level):
    """Cache da busca no painel para não bater na API repetidamente."""
    client = get_openalex_client()
    # Normalização e busca
    normalized_query = client.normalize_query(query)
    raw_articles = client.search_articles(normalized_query, limit)
    
    # Processamento leve dos conceitos (extração) para evitar transportar objetos pesados
    # Se possível, faça a filtragem de score/level aqui e retorne apenas o necessário
    return raw_articles

# ==================== FRAGMENTS PARA ETAPA 2 (NÍVEL DO MÓDULO) ====================

@st.fragment
def render_etapa_2a(d, r):
    """Fragment estável para etapa 2a - Visualização do Grafo"""
    st.header("🕸️ 2. Grafo de conceitos")
    st.caption("Etapa 2: Explore o grafo e o glossário antes de selecionar os conceitos")

    with st.container(border=True):
        st.caption("📋 **Dados do Projeto**")
        st.write(f"**🎯 Tema:** {d['tema']}")
        st.write(f"**❓ Questão:** {d['questao']}")
        st.write(f"**🔑 Palavras-chave:** {d['palavras_chave']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Artigos Analisados", r.get('articles_count', 0))
    col2.metric("🧩 Conceitos no Grafo", r['graph_stats']['nodes'])
    col3.metric("🔗 Conexões", r['graph_stats']['edges'])

    col_grafo, col_glossario = st.columns([1, 1])

    with col_grafo:
        st.subheader("🕸️ Grafo de Coocorrências")
        if r.get('visualization_path'):
            st.image(r['visualization_path'], width="stretch")
        else:
            st.warning("⚠️ Visualização não disponível")

    with col_glossario:
        st.subheader("📖 Glossário de Conceitos")
        with st.container(height=400):
            st.markdown(r.get('glossary', '⚠️ Glossário não disponível'))

    st.divider()
    
    with st.expander("📚 Por que limitamos a 9 conceitos?", expanded=False):
        st.markdown("""
Limitar a exibição de nós em grafos de palavras-chave reduz a sobrecarga cognitiva, 
permitindo que o usuário identifique relações semânticas sem exceder sua capacidade 
limitada de memória de trabalho.

**Referência:**

MILLER, George A. The magical number seven, plus or minus two: some limits on our 
capacity for processing information. *Psychological Review*, Washington, v. 63, n. 2, 
p. 81-97, 1956. DOI: [https://doi.org/10.1037/h0043158](https://doi.org/10.1037/h0043158).
        """)

    st.info("""
    💡 **Próximo passo:** Observe atentamente o grafo e o glossário acima. 
    Na próxima etapa, você selecionará os conceitos mais relevantes para sua pesquisa.
    Essa seleção será usada para gerar uma interpretação personalizada do grafo.
    """)

@st.fragment
def render_etapa_2b(d, r):
    """Fragment estável para etapa 2b - Seleção de Conceitos"""
    primeiro_nome = d['nome'].split()[0]
    st.header("🎯 3. Seleção de Conceitos")
    st.caption("Etapa 3: Escolha os conceitos mais relevantes para sua pesquisa")

    st.markdown(f"""
    ### {primeiro_nome}, quais conceitos do grafo são mais relevantes para seu projeto?

    Considerando seu tema **"{d['tema']}"**, selecione os conceitos que você considera 
    mais importantes para o delineamento do escopo da sua pesquisa.

    *Selecione pelo menos 1 conceito para continuar.*
    """)

    with st.expander("🕸️ Grafo de Referência", expanded=False):
        if r.get('visualization_path'):
            st.image(r['visualization_path'], width="stretch")

@st.fragment
def render_etapa_2c(d, r, selected):
    """Fragment estável para etapa 2c - Relatório"""
    st.header("📋 4. Relatório")
    st.caption("Etapa 4: Interpretação baseada nos conceitos que você selecionou")

    st.success(f"✅ **Conceitos selecionados:** {', '.join(selected)}")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**👤 Aluno:** {d['nome']}")
            st.write(f"**📧 E-mail:** {d['email']}")
        with col2:
            st.write(f"**📅 Data:** {d['timestamp']}")
            st.write(f"**💭 Confiança:** {d['confianca']}")

    with st.container(border=True):
        st.write(f"**🎯 Tema:** {d['tema']}")
        st.write(f"**❓ Questão:** {d['questao']}")
        st.write(f"**🔑 Palavras-chave:** {d['palavras_chave']}")

    st.subheader("📋 Avaliação do Projeto")
    with st.container(border=True):
        st.markdown(r.get('full_report', '⚠️ Avaliação não disponível'))

    st.subheader("💡 Interpretação Personalizada do Grafo")
    with st.container(border=True):
        interpretation = st.session_state.get('personalized_interpretation', '')
        if interpretation:
            st.markdown(interpretation)
        else:
            st.markdown(r.get('graph_interpretation', '⚠️ Interpretação não disponível'))

    st.subheader("🕸️ Grafo de Coocorrências")
    if r.get('visualization_path'):
        st.image(r['visualization_path'], width="stretch")

    with st.expander("📖 Glossário de Conceitos", expanded=False):
        st.markdown(r.get('glossary', '⚠️ Glossário não disponível'))

    st.subheader("🔑 Sugestões de Palavras-chave")
    suggested_kws = st.session_state.get('suggested_keywords', [])
    if suggested_kws:
        for idx, kw in enumerate(suggested_kws):
            with st.container(border=True, key=f"kw_mod_{idx}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{kw.get('term_en', 'N/A')}**")
                    st.caption(f"({kw.get('term_pt', 'N/A')})")
                with col2:
                    st.write(kw.get('description', ''))
    else:
        st.info("Sugestões de palavras-chave não disponíveis")

# ==================== SIDEBAR FIXO ====================
with st.sidebar:
    
    logo_path = "assets/delineia_logo.png"
    img_base64 = get_base64_image(logo_path)
    
    if img_base64:
        html_logo = f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_base64}" style="width: 180px; max-width: 100%;">
            <h1 style="font-size: 24px; margin-top: 10px; margin-bottom: 0;">📋 O que é Delinéia?</h1>
        </div>
        """
        st.markdown(html_logo, unsafe_allow_html=True)
    else:
        # Fallback se a imagem não for encontrada 
        
        st.title("📋 O que é Delinéia?")
    
    st.markdown("---")
  
    with st.expander("Sobre o Delinéia"):
        st.markdown("""
           O Delinéia é um sistema de apoio ao delineamento do escopo temático de projetos de pesquisa no ensino superior e foi desenvolvido como parte de uma tese de doutorado em Informática na Educação. O sistema combina Inteligência Artificial Generativa (Gemini Pro) com análise bibliométrica de coocorrência de palavras a partir de buscas contextuais na base OpenAlex. A proposição visa auxiliar estudantes de graduação e de pós-graduação no esboço de seus projetos de pesquisa.
            """)
    
    with st.expander("Abordagem Interdisciplinar"):
        st.markdown("""
            Este projeto situa-se na colaboração entre os campos da Informática na Educação e da Ciência da Informação, explorando como tecnologias de IA podem apoiar processos de aprendizagem científica no ensino superior.        
        """)
 
    with st.expander("Autoria"):
        st.markdown("""
            **Desenvolvimento:** 
            - Rafael Antunes dos Santos  
            
            **Orientador:** 
            - Prof. Dr. Eliseo Berni Reategui
            
            **Nível:** 
            - Doutorado
            
            **Instituição:**             
            - Universidade Federal do Rio Grande do Sul (UFRGS) 
            - Centro Interdisciplinar de Novas Tecnologias na Educação (Cinted)
            - Programa de Pós-Graduação em Informática na Educação (PPGIE)
                              
            **Formação Anterior:**
            - Mestre em Comunicação e Informação pela UFRGS (PPGCOM)  
            - Bacharel em Biblioteconomia pela UFRGS (DCI/FABICO) - CRB10/1898
        
            **Lattes:** 
            - [http://lattes.cnpq.br/5228660998907867](http://lattes.cnpq.br/5228660998907867)

            **ORCID:** 
            - https://orcid.org/0000-0002-1529-9063 
            
            **Contato:**
            - rafael.antunes@ufrgs.br
            - rderafa@gmail.com

            **Github:**
            - https://github.com/rderafa-rgb        
            """)
    
    with st.expander("Trilha de Aprendizagem"):
        st.markdown("""
            - **Delineascópio:**
              - Trilha gamificada
              - Visualização de conceitos
              - Glossário
              - Seleção de conceitos
              - Avaliação do projeto:
                - Palavras-chave
                - Questão de pesquisa
              - Interpretação personalizada
              - Sugestão de palavras-chave
              - Sugestão de chaves de busca
              - Relatório em PDF        
            - **Interação:** 
              - Grafo completo
              - Visualização dinâmica 
              - Inclusão e exclusão de nós
              - Exportação de redes
              - Construtor de chaves de busca
            - **Histórico:** 
              - Comparação entre grafos
              - Abstração hierárquica
                - Conceitos incluídos
                - Conceitos excluídos
                - Núcleo estável
              - Análise Pedagógica da Mudança
              - Relatório em PDF
            - **Painel:** 
              - Busca de dados com OpenAlex:
                - Artigos: *métricas de artigos e metadados únicos*
                - Conceitos: *métricas de conceitos, nuvem de palavras e lei de Zipf*
                - Coocorrências: *métricas de pares associados e similaridade*
                - Grafo: *visualização estática e clusterização*
                - Mapa Temático: *posição estratégica do cluster*
                - Estatísticas: *resumo breve*
                - Exportação: *dados em JSON, CSV, GraphML, .net, .xlsx, BibTeX, .ris e Zip File*     
            """)
    
    with st.expander("Tecnologias"):
        st.markdown("""
            - OpenAlex API
            - Google Gemini AI 2.5 Pro API
            - Anthropic Claude Opus 4.5
            - Python
            - HuggingFace
            - Streamlit
            - NetworkX
            - Plotly
            - PyVis
            - ReportLab
            - GraphViz
            - JavaScript
            - CSS
            - HTML

            *Versão*
            - Delinéia I (17 nov. 2025)        
            """)
    
    with st.expander("Agradecimentos"):
        st.markdown("""
            Ao **Orientador** 
            - Eliseo Berni Reategui
                     
            Aos **Professores** 
            - Alexandra Lorandi 
            - Alexandre Ribas Semeler
            - Dante Augusto Couto Barone
            - Elisa Boff
            - Fernando Becker
            - Gabriela Trindade Perry
            - Ida Regina Chitto Stumpf 
            - Leandro Krug Wives 
            - Marcus Vinicius de Azevedo Basso
            - Maria de Fátima Santos Maia
            - Milton Antonio Zaro
            - Patrícia Fernanda da Silva
            - Rafael Port da Rocha
            - Regina Helena Van der Laan
            - Renato Ventura Bayan Henriques
            - Rosa Maria Vicari
            - Samile Andréa de Souza Vanz
            - Sérgio Roberto Kieling Franco
            - Sonia Elisa Caregnato
            - Vanessa Soares Maurente
            
            Aos **Colegas** do grupo de pesquisa GTech.Edu.  
            
            À **CAPES**, pela concessão de bolsa de estudos.
            """)

    with st.expander("Publicações"):
        st.markdown("""
            *Artigos relacionados:*
            - SANTOS, R.A.; REATEGUI, E.B. Uso de inteligência artificial generativa e análise de palavras-chave para apoiar o planejamento de projetos de pesquisa no ensino superior. *RELATEC: Revista Latinoamericana de Tecnología Educativa*, v.24, n.2, p.87–104, 2025. Doi: https://doi.org/10.17398/1695-288X.24.2.87.
            - SANTOS, R.A.; REATEGUI, E.B.; CAREGNATO, S.E. Análise de coocorrência de palavras na pesquisa brasileira em HIV/AIDS indexada na Web of Science no período 1993-2020. *Informação & Informação*, v.27, n.2, p.248–273, 2022. Doi: https://doi.org/10.5433/1981-8920.2022v27n2p248. Disponível em: https://ojs.uel.br/revistas/uel/index.php/informacao/article/view/45335.        
           
            *Colaboração em pesquisas:*
            - REATEGUI, E.B.; BIGOLIN, M.; CARNIATO, M.; SANTOS, R.A. Evaluating the performance of SOBEK text mining keyword extraction algorithm. In: HOLZINGER, A. et al. (ed.). *Machine Learning and Knowledge Extraction*: CD-MAKE 2022. Cham: Springer, 2022. p.233–243. (Lecture Notes in Computer Science, 13480. Doi: https://doi.org/10.1007/978-3-031-14463-9_15.
            - SEMELER, A.R.; SANTOS, R.A.; SOARES, K.U. Análise de domínio aplicada aos estudos fronteiriços brasileiros: metadados de publicações científicas de acesso aberto extraídos da plataforma Lattes. In: *ANUÁRIO Unbral das fronteiras brasileiras*: volume 1. Porto Alegre: Instituto de Geociências, 2014. p.37–65.
           """)

    with st.expander("Advertência"):
        st.markdown("""
            *Atenção:*
            - IAs podem cometer erros.
            - IAs podem produzir viés. Potenciais riscos sociais ou comportamentais.
            - IAs podem gerar alucinações ou insegurança científica. 
           """) 

    st.markdown("---") # Linha divisória

    # LICENÇA CREATIVE COMMONS (Formatada em HTML)
    html_cc = """
    <div style="text-align: center; font-size: 0.85em; color: #666;">
        <p>
            <a href="https://huggingface.co/spaces/RafaelAntunes123/Delineia" target="_blank" style="text-decoration:none; color:#3366cc; font-weight:bold;">Delinéia</a> 
            © 2025 by 
            <a href="https://github.com/rderafa-rgb" target="_blank" style="text-decoration:none; color:#3366cc;">Rafael Antunes</a>
        </p>
        <p>Licensed under: <br>
        <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" style="text-decoration:none; color:#3366cc;">CC BY-NC-ND 4.0 International</a>
        </p>
        <p>
            <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" style="height:22px; margin:2px;">
            <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" style="height:22px; margin:2px;">
            <img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" style="height:22px; margin:2px;">
            <img src="https://mirrors.creativecommons.org/presskit/icons/nd.svg" style="height:22px; margin:2px;">
        </p>
    </div>
    """
    st.markdown(html_cc, unsafe_allow_html=True)

# ==================== BIBLIOTECA DE GÊNERO ====================

def genero_texto(masc: str, fem: str, neutro: str = None) -> str:
    """
    Retorna texto adequado ao gênero selecionado pelo usuário.
    
    Uso: genero_texto("Explorador", "Exploradora", "Explorador(a)")
    """
    genero = st.session_state.get('genero', 'Neutro')
    
    if genero == 'Feminino':
        return fem
    elif genero == 'Masculino':
        return masc
    else:
        # Neutro: usa versão com (a) ou fallback para masculino
        return neutro if neutro else f"{masc}(a)"


def g(masc: str, fem: str, neutro: str = None) -> str:
    """Alias curto para genero_texto()."""
    return genero_texto(masc, fem, neutro)

# ==================== RODAPÉ INSTITUCIONAL ====================
def rodape_institucional():
    import base64
    import os

    def get_img_as_base64(path):
        try:
            with open(path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except FileNotFoundError:
            return ""

    # Ajuste os nomes aqui se necessário
    img_ufrgs = get_img_as_base64("assets/ufrgs_logo.png")
    img_cinted = get_img_as_base64("assets/cinted_logo.png") 
    img_ppgie = get_img_as_base64("assets/ppgie_logo.png")

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # HTML Alinhado à esquerda para evitar bugs de Markdown
    html_code = f"""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: sans-serif;">
<div style="display: flex; gap: 30px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; justify-content: center;">
<img src="data:image/png;base64,{img_ufrgs}" style="height: 85px; width: auto; opacity: 0.9;">
<img src="data:image/png;base64,{img_cinted}" style="height: 85px; width: auto; opacity: 0.9;">
<img src="data:image/png;base64,{img_ppgie}" style="height: 105px; width: auto; opacity: 0.9;">
</div>
<div style="text-align: center; color: #666; font-size: 0.85rem; line-height: 1.6;">
<p style="margin-bottom: 10px;">
<b>Delinéia</b> - Sistema de Apoio ao Delineamento de Escopo Temático<br>
Pesquisa de Doutorado - PPGIE / UFRGS
</p>
<hr style="width: 50%; margin: 10px auto; border: 0; border-top: 1px solid #ddd;">
<p style="font-size: 0.75rem; color: #888;">
Delinéia © 2025 by Rafael Antunes is licensed under 
<a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" style="color: #555; font-weight: bold; text-decoration: underline;">
CC BY-NC-ND 4.0
</a>.<br>
To view a copy of this license, visit 
<a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" style="color: #888; text-decoration: none;">
https://creativecommons.org/licenses/by-nc-nd/4.0/
</a>
</p>
</div>
</div>
"""
    st.markdown(html_code, unsafe_allow_html=True)

# ==================== GOOGLE SHEETS CONFIG ====================
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1z5Btw5LLZhwdvBDstd8L_LYf2A0tL0Gjc_6KJPiyAE4/edit?gid=1488517131#gid=1488517131"
ABA_FORMULARIO_INICIAL = "formulario_inicial"
ABA_RESULTADOS_PIPELINE = "resultados_pipeline"
ABA_FORMULARIO_AVALIACAO = "formulario_avaliacao"


@st.cache_resource(show_spinner=False)
def conectar_google_sheets():
    """
    Conecta ao Google Sheets usando credenciais do Streamlit Secrets.
    Compatível com Streamlit Cloud e HuggingFace Spaces.
    """
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    try:
        # Ler credenciais dos secrets
        google_creds = st.secrets["google_credentials"]
        
        # Converter para dict (compatível com Streamlit Cloud e HuggingFace)
        if isinstance(google_creds, str):
            # HuggingFace: secret é string JSON
            import json
            creds_dict = json.loads(google_creds)
        elif hasattr(google_creds, 'to_dict'):
            # Streamlit Cloud: objeto AttrDict
            creds_dict = google_creds.to_dict()
        else:
            # Fallback: tentar converter direto
            creds_dict = dict(google_creds)
        
        # Corrigir private_key (\\n → \n)
        if "private_key" in creds_dict:
            pk = creds_dict["private_key"]
            if "\\n" in pk:
                pk = pk.replace("\\n", "\n")
            creds_dict["private_key"] = pk
        
        # Criar credenciais
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=SCOPES
        )
        
        # Autorizar e abrir planilha
        client = gspread.authorize(creds)
        sheet = client.open_by_url(GOOGLE_SHEETS_URL)
        
        print("✅ Conexão com Google Sheets estabelecida!")
        return sheet
        
    except Exception as e:
        st.error(f"❌ Erro ao conectar Google Sheets: {e}")
        import traceback
        print(f"Detalhes do erro: {traceback.format_exc()}")
        return None

# ==================== HISTÓRICO DE GRAFOS (SHEETS) ====================
def salvar_grafo_historico(id_usuario, form_data, result):
    """
    Salva histórico com estrutura CLARA: Metadados, Nós e Arestas separados por cabeçalhos.
    """
    try:
        sheet = conectar_google_sheets()
        if not sheet: return False
        
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        safe_id = id_usuario.split('_')[-1] if '_' in id_usuario else id_usuario[-8:]
        tab_title = f"G_{safe_id}_{timestamp}"
        
        G = result.get('graph')
        freq = result.get('concept_freq', {})
        
        # Recupera metadados ricos (Score/Level) usando a função global
        raw_articles = result.get('raw_articles', [])
        if 'extract_concept_metadata' in globals():
            concept_meta = extract_concept_metadata(raw_articles)
        else:
            # Fallback seguro caso a função não esteja no topo
            concept_meta = {} 
        
        # 1. BLOCO DE METADADOS
        context_data = [
            ["---METADATA---", "Valor", "", ""],
            ["id_usuario", id_usuario, "", ""],
            ["timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "", ""],
            ["aluno_nome", form_data.get('nome', ''), "", ""],
            ["aluno_tema", form_data.get('tema', ''), "", ""],
            ["aluno_questao", form_data.get('questao', ''), "", ""],
            ["aluno_confianca_ini", form_data.get('confianca', ''), "", ""],
            ["aluno_google_academico", form_data.get('google_academico', ''), "", ""],
            ["pipeline_string", result.get('search_string', ''), "", ""],
            ["pipeline_artigos_total", result.get('articles_count', 0), "", ""],
        ]
        
        # 2. BLOCO DE NÓS
        # Marcador de seção + Cabeçalho explícito na linha seguinte
        nodes_section = [["---NODES---", "", "", ""]] 
        nodes_header = [["Id", "Freq", "Score", "Level"]]
        
        nodes_data = []
        for node in G.nodes():
            m = concept_meta.get(node, {})
            nodes_data.append([
                node, 
                freq.get(node, 1), 
                f"{m.get('score', 0):.4f}", 
                f"{m.get('level', 0):.1f}"
            ])
            
        # 3. BLOCO DE ARESTAS (GRAFO REAL)
        # Marcador de seção + Cabeçalho explícito (Source, Target...)
        edges_section = [["---EDGES---", "", "", ""]]
        edges_header = [["source", "target", "weight", "salton"]]
        
        edges_data = []
        for u, v in G.edges():
            weight = G[u][v].get('weight', 1)
            f_u = freq.get(u, 1)
            f_v = freq.get(v, 1)
            salton = weight / np.sqrt(f_u * f_v) if f_u > 0 and f_v > 0 else 0
            
            # Aqui garantimos que source e target estão nas colunas A e B
            edges_data.append([u, v, weight, f"{salton:.4f}"])
        
        # Montagem Final: Metadata -> Nodes -> Edges
        full_payload = context_data + nodes_section + nodes_header + nodes_data + edges_section + edges_header + edges_data
        
        worksheet = sheet.add_worksheet(title=tab_title, rows=len(full_payload)+20, cols=5)
        worksheet.update(full_payload)
        
        print(f"✅ Grafo salvo corretamente: {tab_title}")
        return True

    except Exception as e:
        st.error(f"❌ Erro ao salvar histórico: {str(e)}")
        print(f"Erro log: {e}")
        return False

def enviar_formulario_inicial(form_data, existing_id=None):
    """Envia dados do formulário inicial para Google Sheets"""
    try:
        sheet = conectar_google_sheets()
        
        if not sheet:
            return None
        
        worksheet = sheet.worksheet(ABA_FORMULARIO_INICIAL)
                
        # Usa ID existente se houver, senão gera novo
        if existing_id:
            id_usuario = existing_id
        else:
            id_usuario = f"user_{uuid.uuid4().hex[:8]}"
        
        # Preparar linha
        row = [
            id_usuario,
            form_data['timestamp'],
            form_data['nome'],
            form_data['email'],
            form_data['tema'],
            form_data['questao'],
            form_data['palavras_chave'],
            form_data.get('google_academico', ''),
            form_data.get('confianca', '')
        ]
        
        worksheet.append_row(row, value_input_option='RAW')
        return id_usuario
        
    except Exception as e:
        st.error(f"❌ Erro ao enviar formulário: {e}")
        return None

def enviar_resultados_pipeline(id_usuario, result, tempo_segundos):
    """Envia resultados do pipeline para Google Sheets"""
    try:
        sheet = conectar_google_sheets()
        if sheet is None:
            return False
        
        worksheet = sheet.worksheet(ABA_RESULTADOS_PIPELINE)
        
        # Preparar linha
        top_conceitos_str = ",".join(result.get('top_concepts', [])[:9])
        
        # Extrair termos sugeridos
        termos_sugeridos = ""
        suggested_kws = result.get('suggested_keywords', [])
        if suggested_kws:
            termos_sugeridos = ", ".join([
                f"{kw.get('term_pt', '')} ({kw.get('term_en', '')})" 
                for kw in suggested_kws
            ])
        
        row = [
            id_usuario,
            datetime.now().strftime("%d/%m/%Y às %H:%M"),
            result.get('search_string', ''),
            termos_sugeridos,
            result.get('search_objective', ''),
            result.get('articles_count', 0),
            top_conceitos_str,
            result['graph_stats']['nodes'],
            result['graph_stats']['edges'],
            result['graph_stats'].get('density', 0),
            round(tempo_segundos, 2)
        ]
        
        worksheet.append_row(row, value_input_option='RAW')
        return True
        
    except Exception as e:
        st.error(f"❌ Erro ao enviar resultados: {e}")
        return False

def atualizar_termos_sugeridos(id_usuario, suggested_keywords):
    """Atualiza coluna termos_sugeridos no Google Sheets"""
    try:
        sheet = conectar_google_sheets()
        if sheet is None:
            return False
        
        worksheet = sheet.worksheet(ABA_RESULTADOS_PIPELINE)
        
        # Formatar termos
        termos_str = ", ".join([
            f"{kw.get('term_pt', '')} ({kw.get('term_en', '')})" 
            for kw in suggested_keywords
        ]) if suggested_keywords else ""
        
        # Encontrar linha do usuário (coluna A = id_usuario)
        try:
            cell = worksheet.find(id_usuario)
            if cell:
                # Atualizar coluna D (termos_sugeridos)
                worksheet.update_cell(cell.row, 4, termos_str)
                return True
        except:
            pass
        
        return False
        
    except Exception as e:
        # Silencioso - não crítico
        return False

def enviar_formulario_avaliacao(id_usuario, avaliacao_data):
    """Envia avaliação do usuário para Google Sheets"""
    try:
        sheet = conectar_google_sheets()
        if sheet is None:
            return False
        
        worksheet = sheet.worksheet(ABA_FORMULARIO_AVALIACAO)
        
        # Calcular tempo total
        tempo_total = 0
        if 'timestamp_formulario_inicial' in st.session_state:
            tempo_total = round(time_module.time() - st.session_state.timestamp_formulario_inicial, 2)
        
        # Preparar linha
        row = [
            id_usuario,
            datetime.now().strftime("%d/%m/%Y às %H:%M"),
            avaliacao_data.get('q1', ''),
            avaliacao_data.get('q2', ''),
            avaliacao_data.get('q3', ''),
            avaliacao_data.get('q4', ''),
            avaliacao_data.get('q5', ''),
            avaliacao_data.get('q6', ''),
            avaliacao_data.get('q7', ''),
            avaliacao_data.get('q8', ''),
            avaliacao_data.get('q9', ''),
            avaliacao_data.get('q10', ''),
            avaliacao_data.get('q11', ''),
            avaliacao_data.get('q12', ''),
            avaliacao_data.get('q13', ''),
            avaliacao_data.get('q14', ''),
            avaliacao_data.get('q15', ''),
            avaliacao_data.get('q16', ''),
            avaliacao_data.get('q17', ''),
            avaliacao_data.get('q18', ''),
            avaliacao_data.get('q19', ''),
            avaliacao_data.get('q20', ''),
            avaliacao_data.get('nps', 0),
            avaliacao_data.get('nps_category', ''),
            avaliacao_data.get('q22', ''),
            avaliacao_data.get('q23', ''),
            avaliacao_data.get('q24', ''),
            avaliacao_data.get('q25', ''),
            avaliacao_data.get('q26', ''),
            avaliacao_data.get('q27', ''),
            avaliacao_data.get('q28', ''),
            avaliacao_data.get('q29', ''),
            avaliacao_data.get('q30', ''),
            'Sim' if avaliacao_data.get('tcle_aceite', False) else 'Não',
            'Sim' if avaliacao_data.get('tcle_rejeita', False) else 'Não',
            'Sim' if avaliacao_data.get('aceite_continuidade', False) else 'Não',
            'Sim' if avaliacao_data.get('rejeita_continuidade', False) else 'Não',
            ",".join(st.session_state.get('badges', [])),
            tempo_total
        ]
        
        worksheet.append_row(row, value_input_option='RAW')
        return True
        
    except Exception as e:
        st.error(f"❌ Erro ao enviar avaliação: {e}")
        return False

def gerar_analise_evolucao(metrics, nome_aluno):
    """
    Usa o Gemini para interpretar a mudança entre dois delineamentos.
    """
    import google.generativeai as genai
    
    # Prepara as listas (limitando a 50 termos para não estourar o prompt com ruído)
    abandonados = ", ".join(metrics['exclusivos_antigos'][:50])
    novos = ", ".join(metrics['exclusivos_novos'][:50])
    mantidos = ", ".join(metrics['comuns'][:30])
    jaccard = f"{metrics['jaccard']*100:.1f}%"
    
    prompt = f"""
    Atue como um Orientador Acadêmico Sênior e Especialista em Bibliometria.
    
    O aluno {nome_aluno} realizou dois delineamentos de pesquisa (buscas bibliográficas) em momentos diferentes.
    Sua tarefa é analisar a EVOLUÇÃO do pensamento dele baseando-se na mudança do vocabulário dos grafos.
    
    DADOS DA MUDANÇA:
    - Similaridade entre os momentos (Jaccard): {jaccard} (quanto menor, maior a mudança).
    - O que ele ABANDONOU (Termos que saíram): {abandonados}
    - O que ele ADOTOU (Novos termos): {novos}
    - Núcleo Estável (O que ficou): {mantidos}
    
    ANÁLISE SOLICITADA (Seja direto, encorajador e analítico):
    1. **Diagnóstico da Mudança:** O escopo afunilou (ficou mais específico)? Expandiu (ficou mais genérico)? Ou mudou completamente de área (pivô)?
    2. **Análise dos Termos:** Cite exemplos específicos. "Ao trocar X por Y, nota-se que..."
    3. **Nível de Maturidade:** A entrada de termos novos sugere uma pesquisa mais madura/técnica ou ainda exploratória?
    4. **Veredito:** Em uma frase, defina essa evolução.
    
    Não use introduções genéricas. Vá direto ao ponto. Use Markdown.
    """
    
    try:
        # Usa o modelo que já está configurado no seu app
        model = genai.GenerativeModel('models/gemini-2.5-pro') # Ou o modelo que você estiver usando
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro ao gerar análise: {str(e)}"

# ==================== FUNÇÃO DE ANÁLISE DE ZIPF =================
def analyze_zipf(frequency_data):
    """
    Analisa a distribuição de frequências segundo a Lei de Zipf

    Args:
        frequency_data: Lista de tuplas (palavra, frequência) ordenada por frequência

    Returns:
        dict com métricas e dados para plotagem
    """
    # Extrair frequências
    frequencies = [freq for _, freq in frequency_data]

    # Criar ranks (1, 2, 3, ...)
    ranks = np.arange(1, len(frequencies) + 1)

    # Converter para arrays numpy
    ranks_array = np.array(ranks)
    freq_array = np.array(frequencies)

    # Aplicar log para análise linear
    log_ranks = np.log10(ranks_array)
    log_freqs = np.log10(freq_array)

    # Regressão linear no espaço log-log
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)

    # Calcular R²
    r_squared = r_value ** 2

    # Gerar linha de tendência
    trend_line = 10 ** (slope * log_ranks + intercept)

    # Interpretação
    if r_squared > 0.90:
        interpretation = "✅ Forte aderência à Lei de Zipf"
        quality = "excelente"
    elif r_squared > 0.75:
        interpretation = "⚠️ Aderência moderada à Lei de Zipf"
        quality = "boa"
    else:
        interpretation = "❌ Fraca aderência à Lei de Zipf"
        quality = "baixa"

    # Análise da inclinação
    if -1.2 < slope < -0.8:
        slope_interpretation = "próximo ao ideal (-1.0)"
    elif slope < -1.2:
        slope_interpretation = "vocabulário mais concentrado que o esperado"
    else:
        slope_interpretation = "vocabulário mais disperso que o esperado"

    return {
        'ranks': ranks_array,
        'frequencies': freq_array,
        'log_ranks': log_ranks,
        'log_freqs': log_freqs,
        'trend_line': trend_line,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'interpretation': interpretation,
        'quality': quality,
        'slope_interpretation': slope_interpretation
    }

# ==================== ESTADOS DA SESSÃO ====================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'resultado' not in st.session_state:
    st.session_state.resultado = None
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None
if 'dashboard_query' not in st.session_state:
    st.session_state.dashboard_query = ""
if 'avaliacao_completa' not in st.session_state:
    st.session_state.avaliacao_completa = False
if 'badges' not in st.session_state:
    st.session_state.badges = []
if 'play_video' not in st.session_state:
    st.session_state.play_video = False
if 'open_prologo' not in st.session_state:
    st.session_state.open_prologo = False
if 'selected_concepts' not in st.session_state:
    st.session_state.selected_concepts = []
if 'interpretation_generated' not in st.session_state:
    st.session_state.interpretation_generated = False
if 'personalized_interpretation' not in st.session_state:
    st.session_state.personalized_interpretation = None
if 'suggested_keywords' not in st.session_state:
    st.session_state.suggested_keywords = []
if 'suggested_strings' not in st.session_state:
    st.session_state.suggested_strings = {}
if 'sub_step' not in st.session_state:
    st.session_state.sub_step = 'a'  # 'a', 'b', 'c'

# ==================== FUNÇÕES AUXILIARES ====================
def add_badge(badge_name: str) -> bool:
    """
    Adiciona badge, removendo versões anteriores (de outro gênero) do mesmo badge.
    Identifica o badge pelo ícone (primeiro caractere).
    """
    # Se o badge exato já existe, não faz nada
    if badge_name in st.session_state.badges:
        return False
        
    # Identificar o ícone (ex: 🎯, 🔬)
    icone = badge_name.split(' ')[0]
    
    # Remover versões antigas desse mesmo badge (ex: remove "Explorador" se for entrar "Exploradora")
    st.session_state.badges = [b for b in st.session_state.badges if not b.startswith(icone)]
    
    # Adicionar a nova versão correta
    st.session_state.badges.append(badge_name)
    return True

def process_openalex_dataframe(articles):
    """Transforma a lista bruta de artigos em um DataFrame limpo para exibição."""
    data = []
    for art in articles:
        # Pega o primeiro autor (ou 'N/A')
        authors = art.get('authorships', [])
        first_author = authors[0].get('author', {}).get('display_name', 'N/A') if authors else 'N/A'
        
        # Pega top 3 conceitos
        concepts = [c.get('display_name', '') for c in art.get('concepts', [])]
        top_concepts = ", ".join(concepts[:3])
        
        data.append({
            'Título': art.get('title', 'Sem título'),
            'Ano': art.get('publication_year', art.get('year', 'N/A')),
            'Autor (1º)': first_author,
            'Top Conceitos': top_concepts,
            'Citações': art.get('cited_by_count', 0),
            'DOI/URL': art.get('doi', art.get('url', ''))
        })
    return pd.DataFrame(data)

def calculate_layout_positions(G: nx.Graph, layout_name: str) -> dict:
    """
    Calcula posições dos nós usando diferentes algoritmos de layout.
    """
    scale = 500
    
    if layout_name == "Kamada-Kawai":
        try:
            pos = nx.kamada_kawai_layout(G, scale=scale)
        except:
            pos = nx.spring_layout(G, scale=scale, seed=42)
    
    elif layout_name == "Circular":
        pos = nx.circular_layout(G, scale=scale)
    
    elif layout_name == "Shell (concêntrico)":
        degrees = dict(G.degree())
        if degrees:
            sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
            n = len(sorted_nodes)
            shells = [
                sorted_nodes[:n//3] if n > 3 else sorted_nodes,
                sorted_nodes[n//3:2*n//3] if n > 3 else [],
                sorted_nodes[2*n//3:] if n > 3 else []
            ]
            shells = [s for s in shells if s]
            pos = nx.shell_layout(G, nlist=shells, scale=scale)
        else:
            pos = nx.shell_layout(G, scale=scale)
    
    elif layout_name == "Spectral":
        try:
            pos = nx.spectral_layout(G, scale=scale)
        except:
            pos = nx.spring_layout(G, scale=scale, seed=42)
    
    elif layout_name == "Random":
        pos = nx.random_layout(G, seed=42)
        pos = {k: (v[0] * scale, v[1] * scale) for k, v in pos.items()}
    
    elif layout_name == "Fruchterman-Reingold":
        k_val = 2 / (len(G.nodes()) ** 0.5) if len(G.nodes()) > 0 else 1
        pos = nx.spring_layout(G, scale=scale, seed=42, k=k_val)
    
    else:
        return None
    
    # Converter numpy.float32 para float nativo
    if pos:
        pos = {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
    
    return pos

def render_interactive_graph_pyvis(G: nx.Graph, selected_concepts: list = None, 
                                    concept_metadata: dict = None, 
                                    layout_positions: dict = None,
                                    enable_physics: bool = True,
                                    height: str = "550px") -> None:
    """
    Renderiza um grafo NetworkX de forma interativa usando PyVis.
    """
    
    if not PYVIS_AVAILABLE:
        st.error("⚠️ PyVis não está instalado. Adicione 'pyvis>=0.3.0' ao requirements.txt")
        return
    
    if G is None or len(G.nodes()) == 0:
        st.warning("Grafo vazio ou não disponível")
        return
    
    nt = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=False
    )
    
    physics_config = "true" if enable_physics else "false"
    
    nt.set_options(f"""
    {{
        "nodes": {{
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {{
                "size": 14,
                "face": "arial"
            }}
        }},
        "edges": {{
            "color": {{
                "color": "#cccccc",
                "highlight": "#10b981"
            }},
            "smooth": {{
                "type": "continuous"
            }}
        }},
        "physics": {{
            "enabled": {physics_config},
            "forceAtlas2Based": {{
                "gravitationalConstant": -60,
                "centralGravity": 0.015,
                "springLength": 120,
                "springConstant": 0.08
            }},
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {{
                "enabled": true,
                "iterations": 200,
                "updateInterval": 25
            }}
        }},
        "interaction": {{
            "hover": true,
            "tooltipDelay": 150,
            "hideEdgesOnDrag": true,
            "zoomView": true,
            "dragView": true
        }}
    }}
    """)
    
    nt.from_nx(G)
    
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    selected = selected_concepts or []
    metadata = concept_metadata or {}
    
    max_freq = max([m.get('freq', 1) for m in metadata.values()]) if metadata else max_degree
    
    for node in nt.nodes:
        node_id = node['id']
        degree = degrees.get(node_id, 1)
        
        meta = metadata.get(node_id, {})
        freq = meta.get('freq', degree)
        score = meta.get('score', 0)
        level = meta.get('level', 0)
        
        node['size'] = 18 + (freq / max_freq) * 35
        
        if layout_positions and node_id in layout_positions:
            pos = layout_positions[node_id]
            node['x'] = pos[0]
            node['y'] = pos[1]
        
        if node_id in selected:
            node['color'] = {
                'background': '#f59e0b',
                'border': '#d97706',
                'highlight': {
                    'background': '#fbbf24',
                    'border': '#b45309'
                }
            }
        else:
            node['color'] = {
                'background': '#10b981',
                'border': '#059669',
                'highlight': {
                    'background': '#34d399',
                    'border': '#047857'
                }
            }
        
        status = "✓ Selecionado" if node_id in selected else ""
        level_desc = ["Geral", "Campo", "Subcampo", "Nicho", "Específico", "Ultra-específico"]
        level_text = level_desc[int(level)] if 0 <= level < len(level_desc) else f"Nível {level:.0f}"
        
        node['title'] = f"""{node_id}
━━━━━━━━━━━━━━━━━━━━
📊 Frequência: {freq} artigos
🎯 Score médio: {score:.2f}
📐 Level: {level:.1f} ({level_text})
🔗 Conexões: {degree}
{status}"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        nt.save_graph(f.name)
        temp_path = f.name
    
    try:
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        h_val = int(height.replace('px', ''))
        components.html(html_content, height=h_val + 50, scrolling=False)
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

def render_tab3_interacao():
    """
    Renderiza a Tab3: Interação com o Grafo
    """
    
    st.title("🔬 Exploração Interativa do Grafo")
    st.caption("Visualize e explore a rede de conceitos de forma dinâmica")
    
    if st.session_state.get('resultado') is None:
        st.info("👈 Complete primeiro o **Delineascópio** para visualizar o grafo interativo.")
        st.markdown("""
        **O que você encontrará aqui:**
        - 🕸️ Grafo interativo (arraste, zoom, clique)
        - 📊 Estatísticas de centralidade
        - 🔍 Filtros dinâmicos por grau e peso
        - 💾 Exportação para Gephi e outros softwares
        """)
        rodape_institucional()
        return
    
    r = st.session_state.resultado
    G = r.get('graph')
    
    if G is None or len(G.nodes()) == 0:
        st.warning("⚠️ Grafo não disponível. Execute o pipeline novamente.")
        rodape_institucional()
        return
    
    # Extrair metadados dos conceitos
    articles = r.get('raw_articles', [])
    concept_metadata = extract_concept_metadata(articles)
    
    selected_concepts = st.session_state.get('selected_concepts', [])
    
    # ==================== CONTROLES DE FILTRO ====================
    with st.expander(f"⚙️ **Filtros do Grafo** ({len(G.nodes())} conceitos disponíveis)", expanded=True):
        
        # Linha 1: Filtros numéricos
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            max_deg_value = max(dict(G.degree()).values()) if G.nodes() else 1
            min_degree = st.slider(
                "Grau mínimo dos nós:",
                min_value=1,
                max_value=max(max_deg_value, 2),
                value=1,
                help="Remove nós com poucas conexões"
            )
        
        with col_f2:
            if G.edges():
                edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
                min_w, max_w = int(min(edge_weights)), int(max(edge_weights))
                min_weight = st.slider(
                    "Peso mínimo das arestas:",
                    min_value=min_w,
                    max_value=max(max_w, min_w + 1),
                    value=min_w,
                    help="Remove conexões fracas"
                )
            else:
                min_weight = 1
        
        with col_f3:
            max_nodes = st.slider(
                "Máximo de nós:",
                min_value=5,
                max_value=min(len(G.nodes()), 100),
                value=min(len(G.nodes()), 50),
                help="Limita visualização aos mais frequentes"
            )
        
        st.divider()
        
        # Linha 2: Seleção de conceitos (INCLUSÃO/EXCLUSÃO)
        all_concepts_sorted = sorted(G.nodes())
        
        col_inc, col_exc = st.columns(2)
        
        with col_inc:
            include_concepts = st.multiselect(
                "✅ Incluir apenas estes conceitos:",
                options=all_concepts_sorted,
                default=[],
                help="Deixe vazio para incluir todos. Se selecionar, mostra APENAS os escolhidos.",
                placeholder="Todos os conceitos (padrão)"
            )
        
        with col_exc:
            exclude_concepts = st.multiselect(
                "❌ Excluir estes conceitos:",
                options=all_concepts_sorted,
                default=[],
                help="Conceitos que serão removidos do grafo.",
                placeholder="Nenhum excluído (padrão)"
            )
        
        st.divider()
        
        # Linha 3: Layout do grafo
        col_layout, col_physics = st.columns(2)
        
        with col_layout:
            layout_option = st.selectbox(
                "🗺️ Layout do grafo:",
                options=[
                    "Força (padrão)",
                    "Kamada-Kawai",
                    "Circular",
                    "Shell (concêntrico)",
                    "Spectral",
                    "Random",
                    "Fruchterman-Reingold"
                ],
                index=0,
                help="Algoritmo de posicionamento dos nós"
            )
        
        with col_physics:
            enable_physics = st.checkbox(
                "⚡ Física ativa",
                value=(layout_option == "Força (padrão)"),
                help="Permite arrastar nós. Desative para layouts fixos.",
                key="chk_physics"
            )
    
    # ==================== APLICAR FILTROS ====================
    G_filtered = G.copy()
    
    if include_concepts:
        nodes_to_keep = set(include_concepts)
        nodes_to_remove = [n for n in G_filtered.nodes() if n not in nodes_to_keep]
        G_filtered.remove_nodes_from(nodes_to_remove)
    
    if exclude_concepts:
        G_filtered.remove_nodes_from(exclude_concepts)
    
    nodes_to_remove = [n for n in G_filtered.nodes() if G_filtered.degree(n) < min_degree]
    G_filtered.remove_nodes_from(nodes_to_remove)
    
    edges_to_remove = [(u, v) for u, v in G_filtered.edges() 
                       if G_filtered[u][v].get('weight', 1) < min_weight]
    G_filtered.remove_edges_from(edges_to_remove)
    
    if len(G_filtered.nodes()) > max_nodes:
        degrees = dict(G_filtered.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G_filtered = G_filtered.subgraph(top_nodes).copy()
    
    isolates = list(nx.isolates(G_filtered))
    G_filtered.remove_nodes_from(isolates)
    
    # ==================== MÉTRICAS ====================
    st.divider()
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("🔵 Nós Visíveis", len(G_filtered.nodes()))
    col_m2.metric("🔗 Arestas Visíveis", len(G_filtered.edges()))
    
    if len(G_filtered.nodes()) > 1:
        density = nx.density(G_filtered)
        col_m3.metric("📐 Densidade", f"{density:.3f}")
    else:
        col_m3.metric("📐 Densidade", "N/A")
    
    col_m4.metric("⭐ Selecionados", len([c for c in selected_concepts if c in G_filtered.nodes()]))
    
    # ==================== GRAFO INTERATIVO ====================
    st.divider()
    
    if len(G_filtered.nodes()) > 0:
        st.subheader("🕸️ Grafo Interativo")
        st.caption("**Arraste** os nós para reorganizar • **Scroll** para zoom • **Clique** para destacar • Nós dourados = selecionados")
        
        layout_positions = calculate_layout_positions(G_filtered, layout_option)
        
        render_interactive_graph_pyvis(
            G_filtered, 
            selected_concepts, 
            concept_metadata, 
            layout_positions,
            enable_physics,
            height="550px"
        )
    else:
        st.warning("⚠️ Nenhum nó atende aos critérios de filtro. Ajuste os controles acima.")
    
    st.divider()
    
    # ==================== ESTATÍSTICAS AVANÇADAS ====================
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        with st.expander("📊 **Centralidade de Grau** (Top 10)", expanded=False):
            if len(G_filtered.nodes()) > 0:
                degree_centrality = nx.degree_centrality(G_filtered)
                sorted_dc = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for i, (node, centrality) in enumerate(sorted_dc, 1):
                    marker = "🟡" if node in selected_concepts else "🟢"
                    st.write(f"{i}. {marker} **{node}**: {centrality:.3f}")
    
    with col_stats2:
        with st.expander("🔀 **Centralidade de Intermediação** (Top 10)", expanded=False):
            if len(G_filtered.nodes()) > 1:
                try:
                    betweenness = nx.betweenness_centrality(G_filtered)
                    sorted_bc = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    for i, (node, centrality) in enumerate(sorted_bc, 1):
                        marker = "🟡" if node in selected_concepts else "🟢"
                        st.write(f"{i}. {marker} **{node}**: {centrality:.3f}")
                except:
                    st.write("Não disponível para este grafo")
            else:
                st.write("Precisa de pelo menos 2 nós")
    
    # ==================== EXPORTAÇÃO ====================
    with st.expander("💾 **Exportar Grafo Filtrado**", expanded=False):
        st.caption("Baixe o grafo com os filtros aplicados para análise em outros softwares.")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            try:
                if 'cache_graphml_interacao' not in st.session_state:
                    graphml_buffer = io.BytesIO()
                    nx.write_graphml(G_filtered, graphml_buffer)
                    graphml_buffer.seek(0)
                    st.session_state.cache_graphml_interacao = graphml_buffer.getvalue()
                
                st.download_button(
                    "📥 GraphML (Gephi)",
                    data=st.session_state.cache_graphml_interacao,
                    file_name="grafo_interativo.graphml",
                    mime="application/xml",
                    width="stretch",
                    help="Para Gephi ou Cytoscape",
                    key="dl_graphml_interacao"
                )
            except Exception as e:
                st.error(f"Erro: {e}")
        
        with col_exp2:
            if 'cache_arestas_csv' not in st.session_state:
                edges_data = ["source,target,weight"]
                for u, v in G_filtered.edges():
                    weight = G_filtered[u][v].get('weight', 1)
                    edges_data.append(f"{u},{v},{weight}")
                st.session_state.cache_arestas_csv = "\n".join(edges_data)
            
            st.download_button(
                "📥 Arestas (CSV)",
                data=st.session_state.cache_arestas_csv,
                file_name="grafo_arestas.csv",
                mime="text/csv",
                width="stretch",
                help="Lista de conexões",
                key="dl_arestas_csv"
            )
        
        with col_exp3:
            if 'cache_nos_csv' not in st.session_state:
                nodes_data = ["node,degree,degree_centrality,selected"]
                degree_cent = nx.degree_centrality(G_filtered) if len(G_filtered.nodes()) > 0 else {}
                
                for node in G_filtered.nodes():
                    deg = G_filtered.degree(node)
                    dc = degree_cent.get(node, 0)
                    sel = "sim" if node in selected_concepts else "não"
                    nodes_data.append(f"{node},{deg},{dc:.4f},{sel}")
                st.session_state.cache_nos_csv = "\n".join(nodes_data)
            
            st.download_button(
                "📥 Nós (CSV)",
                data=st.session_state.cache_nos_csv,
                file_name="grafo_nos.csv",
                mime="text/csv",
                width="stretch",
                help="Lista de conceitos com métricas",
                key="dl_nos_csv"
            )
    
    # ==================== CONSTRUTOR DE CHAVE DE BUSCA ====================
    st.divider()
    st.subheader("🔧 Construtor de Chave de Busca")
    st.caption("Monte sua própria chave de busca selecionando conceitos do grafo e inserindo operadores booleanos")
    
    with st.expander("**Construir Chave Personalizada**", expanded=False):
        
        # Inicializar session_state para o text_area se não existir
        if 'search_key_text' not in st.session_state:
            st.session_state.search_key_text = ""
        if 'collected_terms' not in st.session_state:
            st.session_state.collected_terms = []

        # Conceitos disponíveis (do grafo filtrado ou original)
        available_concepts = sorted(G_filtered.nodes()) if len(G_filtered.nodes()) > 0 else sorted(G.nodes())
        
        # ========== SEÇÃO 1: SELEÇÃO DE CONCEITOS ==========
        st.markdown("**1. Selecione um conceito:**")
        
        selected_concept = st.selectbox(
            "Conceito para formatar:",
            options=[""] + available_concepts,
            index=0,
            help="Escolha um conceito para formatar e adicionar à chave",
            label_visibility="collapsed",
            key="sel_conceito_construtor"
        )
        
        if selected_concept:
            st.divider()
            
            # ========== SEÇÃO 2: FORMATAÇÃO DO TERMO ==========
            st.markdown("**2. Formatação do termo:**")
            
            col_trunc, col_aspas = st.columns(2)
            
            with col_trunc:
                use_truncation = st.checkbox(
                    "Usar truncagem (*)",
                    value=False,
                    help="Adiciona * ao final para recuperar variações",
                    key="chk_truncagem"
                )
            
            with col_aspas:
                use_quotes = st.checkbox(
                    'Usar aspas (" ")',
                    value=True,
                    help="Coloca o termo entre aspas para busca exata",
                    key="chk_aspas"
                )
            
            def format_term(term, truncation=False, quotes=False):
                t = term
                if truncation:
                    words = t.split()
                    if words:
                        words[-1] = words[-1][:4] + "*" if len(words[-1]) > 4 else words[-1] + "*"
                        t = " ".join(words)
                if quotes:
                    t = f'"{t}"'
                return t
            
            formatted_preview = format_term(selected_concept, use_truncation, use_quotes)
            st.code(formatted_preview, language=None)
            
            if st.button("➕ Colecionar termo", width="stretch", type="primary"):
                if formatted_preview not in st.session_state.collected_terms:
                    st.session_state.collected_terms.append(formatted_preview)
                st.rerun()
            
            st.divider()
            
            # ========== SEÇÃO 3: OPERADORES E CONSTRUÇÃO ==========
            st.markdown("**3. Construa a chave de busca:**")
            
            if st.session_state.collected_terms:
                st.caption(f"Termos coletados: {', '.join(st.session_state.collected_terms)}")
            else:
                st.caption("Nenhum termo coletado ainda.")
            
            # Botões de inserção - Operadores Booleanos
            st.markdown("**Operadores booleanos:**")
            col_and, col_or, col_not, col_abre, col_fecha = st.columns(5)
            
            with col_and:
                if st.button("AND", width="stretch", help="Interseção: retorna resultados que contenham TODOS os termos"):
                    st.session_state.search_key_text += " AND "
                    st.rerun()
            
            with col_or:
                if st.button("OR", width="stretch", help="União: retorna resultados que contenham QUALQUER um dos termos"):
                    st.session_state.search_key_text += " OR "
                    st.rerun()
            
            with col_not:
                if st.button("NOT", width="stretch", help="Exclusão: remove resultados que contenham o termo seguinte"):
                    st.session_state.search_key_text += " NOT "
                    st.rerun()
            
            with col_abre:
                if st.button("(", width="stretch", help="Abre parênteses para agrupar termos"):
                    st.session_state.search_key_text += "("
                    st.rerun()
            
            with col_fecha:
                if st.button(")", width="stretch", help="Fecha parênteses"):
                    st.session_state.search_key_text += ")"
                    st.rerun()
            
            # Botões para inserir termos coletados
            if st.session_state.collected_terms:
                st.markdown("**Inserir conceitos:**")
                num_cols = 4
                for i in range(0, len(st.session_state.collected_terms), num_cols):
                    cols = st.columns(num_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(st.session_state.collected_terms):
                            term = st.session_state.collected_terms[idx]
                            display_label = term[:20] + "..." if len(term) > 20 else term
                            with col:
                                if st.button(display_label, key=f"term_btn_{idx}", width="stretch"):
                                    st.session_state.search_key_text += term
                                    st.rerun()
            
            col_limpar, col_limpar_termos = st.columns(2)
            with col_limpar:
                if st.button("🗑️ Limpar chave", width="stretch"):
                    st.session_state.search_key_text = ""
                    st.rerun()
            with col_limpar_termos:
                if st.button("🗑️ Limpar termos coletados", width="stretch"):
                    st.session_state.collected_terms = []
                    st.rerun()
            
            st.divider()
                     
            # ========== SEÇÃO 4: ÁREA DE EDIÇÃO ==========
            st.markdown("**4. Chave de busca:**")
            
            edited_key = st.text_area(
                "Edite sua chave de busca:",
                value=st.session_state.search_key_text,
                height=100,
                help="Você pode editar diretamente este campo.",
                label_visibility="collapsed",
                placeholder="Use os botões acima para construir sua chave..."
            )
            
            if edited_key != st.session_state.search_key_text:
                st.session_state.search_key_text = edited_key
                        
            # Métricas
            col_info1, col_info2 = st.columns(2)
            col_info1.metric("Termos coletados", len(st.session_state.collected_terms))
            col_info2.metric("Caracteres", len(edited_key.strip()))
            
            st.divider()
            
            if st.button("📋 Copiar para o Painel", width="stretch", type="primary", key="btn_copiar_construtor"):
                st.session_state.dashboard_query = edited_key.strip()
                st.session_state.dashboard_query_source = "construtor"
                st.toast("✅ Chave copiada para o Painel!")     
             
    rodape_institucional()

# ==================== ABAS PRINCIPAIS ====================
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Delineascópio", "🔬 Interação", "📜 Histórico", "🔎 Painel"])

# ==================== ABA 1: DELINEASCÓPIO ====================
with tab1:
    
    st.title("🤖 Delinéia - Delineamento de Escopo Temático")
    st.caption("Sistema de apoio ao delineamento de projetos de pesquisa com IA e Bibliometria")

    # Barra de progresso gamificada (5 etapas)
    sub_step = st.session_state.get('sub_step', 'a')
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.session_state.step >= 1:
            st.success("✅ 1. Formulário inicial")
            if f'🎯 {g("Explorador", "Exploradora")}' not in st.session_state.badges:
                add_badge(f'🎯 {g("Explorador", "Exploradora")}')
        else:
            st.info("⏳ 1. Formulário inicial")

    with col2:
        if st.session_state.step >= 2:
            st.success("✅ 2. Grafo de conceitos")
            if f'🔬 {g("Pesquisador", "Pesquisadora")}' not in st.session_state.badges:
                add_badge(f'🔬 {g("Pesquisador", "Pesquisadora")}')
        else:
            st.info("⏳ 2. Grafo de conceitos")

    with col3:
        if st.session_state.step >= 2 and sub_step in ['b', 'c']:
            st.success("✅ 3. Seleção de conceitos")
            if f'🧩 {g("Seletor", "Seletora")}' not in st.session_state.badges:
                add_badge(f'🧩 {g("Seletor", "Seletora")}')
        elif st.session_state.step == 2 and sub_step == 'a':
            st.info("⏳ 3. Seleção de conceitos")
        else:
            st.info("⏳ 3. Seleção de conceitos")

    with col4:
        if st.session_state.step >= 2 and sub_step == 'c':
            st.success("✅ 4. Relatório")
            if f'🏆 {g("Delineador", "Delineadora")}' not in st.session_state.badges:
                add_badge(f'🏆 {g("Delineador", "Delineadora")}')
        elif st.session_state.step > 2:
            st.success("✅ 4. Relatório")
            if f'🏆 {g("Delineador", "Delineadora")}' not in st.session_state.badges:
                add_badge(f'🏆 {g("Delineador", "Delineadora")}')
        else:
            st.info("⏳ 4. Relatório")

    with col5:
        if st.session_state.get('avaliacao_completa', False):
            st.success("✅ 5. Avaliação")
            if f'💎 {g("Avaliador", "Avaliadora")}' not in st.session_state.badges:
                add_badge(f'💎 {g("Avaliador", "Avaliadora")}')
        elif st.session_state.step >= 3:
            st.warning("🔄 5. Avaliação")
        else:
            st.info("⏳ 5. Avaliação")

    # Mostrar badges conquistados
    if st.session_state.badges:
        st.markdown(f"**🏅 Conquistas:** {' '.join(st.session_state.badges)}")

    st.divider()

    # ========== ETAPA 1: FORMULÁRIO INICIAL ==========
    if st.session_state.step == 1:
        st.header("📝 1. Formulário Inicial")

        with st.form("formulario_inicial"):
            st.subheader("👤 Identificação")
            col1, col2 = st.columns(2)

            with col1:
                nome = st.text_input(
                    "Nome completo*",
                    placeholder="Ex: Ana Silva",
                    help="Seu nome completo"
                )

            with col2:
                email = st.text_input(
                    "E-mail*",
                    placeholder="Ex: ana@email.com",
                    help="Seu e-mail para contato"
                )

            # Preferência de gênero para personalização dos textos
            genero = st.radio(
                "Como prefere ser tratado(a) nos textos?",
                options=["Masculino", "Feminino", "Neutro"],
                index=2,  # Neutro como padrão
                horizontal=True,
                help="Usado para personalizar distintivos e textos do relatório"
            )

            st.divider()
            st.subheader("🔬 Projeto de Pesquisa")

            tema = st.text_input(
                "F1.1. Tema da Pesquisa*",
                placeholder="Ex: Jogos como estratégia de ensino e aprendizagem na escola",
                help="Tema principal do seu projeto"
            )

            questao = st.text_area(
                "F1.2. Questão de Pesquisa*",
                placeholder="Ex: Qual a percepção dos professores sobre a eficácia dos jogos como estratégia de ensino e aprendizagem na escola?",
                height=100,
                help="Pergunta principal que você quer responder"
            )

            palavras_chave = st.text_input(
                "F1.3. Palavras-chave* (separadas entre vírgulas)",
                placeholder="Ex: Jogos, Ensino, Aprendizagem, Percepção dos professores",
                help="Separe as palavras-chave por vírgula"
            )

            google_academico = st.text_area(
                "F1.4. Se você fosse pesquisar referências para seu projeto no Google Acadêmico, o que você colocaria no campo de busca?*",
                placeholder="Ex: Uso de jogos na escola",
                help="Campo livre para indicar palavras, frases, etc. que você quer pesquisar",
                height=100
            )

            st.divider()
            st.subheader("💭 Autoavaliação")

            confianca = st.radio(
                "F1.5. Qual seu nível de segurança em relação às palavras-chave escolhidas?",
                options=[
                    "Totalmente seguro",
                    "Seguro",
                    "Neutro",
                    "Inseguro",
                    "Totalmente inseguro"
                ],
                index=2,  # Neutro como padrão
                horizontal=True,
                help="Selecione seu nível de confiança nas palavras-chave escolhidas"
            )
       
            st.divider()

            submitted = st.form_submit_button(
                "🚀 Gerar Relatório de Delineamento",
                type="primary",
                width="stretch"
            )

            if submitted:
                if not all([nome, email, tema, questao, palavras_chave, google_academico]):
                    st.error("⚠️ Por favor, preencha todos os campos obrigatórios (*)")
                else:
                    # Força o reinício da trilha na etapa de visualização (a)
                    st.session_state.sub_step = 'a'
                    
                    # Limpa seleções e sugestões anteriores para não misturar projetos
                    st.session_state.selected_concepts = []
                    st.session_state.suggested_keywords = []
                    st.session_state.suggested_strings = {}
                    st.session_state.interpretation_generated = False
                    st.session_state.personalized_interpretation = None

                    # Salvar dados do formulário
                    st.session_state.form_data = {
                        'nome': nome,
                        'email': email,
                        'genero': genero,
                        'tema': tema,
                        'questao': questao,
                        'palavras_chave': palavras_chave,
                        'confianca': confianca,
                        'google_academico': google_academico,
                        'timestamp': datetime.now(timezone(timedelta(hours=-3))).strftime("%d/%m/%Y às %H:%M")
                    }

                    # Gênero para acesso global (biblioteca de gênero)
                    st.session_state.genero = genero

                    # Verifica se já existe um ID nesta sessão (Continuidade)
                    existing_id = st.session_state.get('id_usuario')

                    # Enviar para Google Sheets e salvar ID (passando o existente)
                    id_usuario = enviar_formulario_inicial(st.session_state.form_data, existing_id)
                  
                    if id_usuario:
                        st.session_state.id_usuario = id_usuario
                        st.session_state.timestamp_formulario_inicial = time_module.time()

                    with st.spinner("🔄 Processando... (aguarde 2-3 minutos)"):
                        try:
                            limpar_memoria()
                            # Inicializar pipeline
                            pipe = ResearchScopePipeline(OPENALEX_EMAIL)

                            # Processar palavras-chave
                            kws = [k.strip() for k in palavras_chave.split(',') if k.strip()]

                            # Executar pipeline
                            tempo_inicio = time_module.time()
                            
                            # Usa a função cacheada
                            st.session_state.resultado = run_cached_pipeline(nome, tema, questao, kws, genero, google_academico)
                            tempo_fim = time_module.time()

                            # Enviar resultados para Google Sheets
                            if 'id_usuario' in st.session_state:
                                enviar_resultados_pipeline(
                                    st.session_state.id_usuario,
                                    st.session_state.resultado,
                                    tempo_fim - tempo_inicio
                                )
                            
                            # Salvar grafo no Google Sheets (histórico)
                            if 'id_usuario' in st.session_state:
                                try:
                                    # CORREÇÃO CRÍTICA: Passando os objetos completos conforme a nova definição
                                    salvar_grafo_historico(
                                        st.session_state.id_usuario,
                                        st.session_state.form_data,
                                        st.session_state.resultado
                                    )
                                except Exception as e:
                                    st.error(f"⚠️ Erro visual ao salvar histórico: {e}")
                                    print(f"⚠️ Erro log: {e}")

                            st.session_state.step = 2
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Erro ao processar: {str(e)}")
                            st.exception(e)
        
        rodape_institucional()

    # ========== ETAPA 2: TRILHA DE APRENDIZAGEM ATIVA ==========
    elif st.session_state.step == 2:
        d = st.session_state.form_data
        r = st.session_state.resultado
        sub_step = st.session_state.get('sub_step', 'a')

        # ========== SUB-ETAPA 2a: VISUALIZAÇÃO DO GRAFO ==========
        if sub_step == 'a':
            if st.button("⬅️ Voltar ao Formulário", key="btn_voltar_form_2a"):
                st.session_state.step = 1
                st.rerun()

            render_etapa_2a(d, r)

            if st.button("Continuar para Seleção de Conceitos ▶️", type="primary", width="stretch", key="btn_continuar_2a"):
                st.session_state.sub_step = 'b'
                st.rerun()

            rodape_institucional()

        # ========== SUB-ETAPA 2b: SELEÇÃO DE CONCEITOS ==========
        elif sub_step == 'b':
            if st.button("⬅️ Voltar ao Grafo", key="btn_voltar_2b"):
                st.session_state.sub_step = 'a'
                st.rerun()

            render_etapa_2b(d, r)

            top_concepts = r.get('top_concepts', [])[:9]
            st.subheader("📋 Conceitos Identificados na Rede")
            cols = st.columns(3)
            selected = []

            for i, concept in enumerate(top_concepts):
                col_idx = i % 3
                with cols[col_idx]:
                    default_value = concept in st.session_state.get('selected_concepts', [])
                    if st.checkbox(concept, value=default_value, key=f"concept_{i}"):
                        selected.append(concept)

            if selected != st.session_state.get('selected_concepts', []):
                st.session_state.selected_concepts = selected

            st.divider()
            num_selected = len(selected)

            if num_selected == 0:
                st.warning("⚠️ Selecione pelo menos 1 conceito para continuar")
            else:
                st.success(f"✅ **{num_selected} conceito(s) selecionado(s):** {', '.join(selected)}")

            col1, col2 = st.columns(2)

            with col2:
                if num_selected >= 1:
                    if st.button("Gerar Relatório de Delineamento ▶️", type="primary", width="stretch", key="btn_gerar_relatorio"):
                        with st.spinner("🔄 Gerando relatório... (aguarde 1-2 minutos)"):
                            from research_pipeline import GeminiQueryGenerator
                            gemini = GeminiQueryGenerator()

                            primeiro_nome = d['nome'].split()[0]
                            tema = d['tema']
                            original_kws = [k.strip() for k in d.get('palavras_chave', '').split(',') if k.strip()]
                            all_concepts = r.get('top_concepts', [])[:9]

                            st.session_state.personalized_interpretation = gemini.generate_contextualized_interpretation(
                                tema, primeiro_nome, selected, all_concepts, genero=d.get('genero', 'Neutro')
                            )

                            st.session_state.suggested_keywords = gemini.generate_keyword_suggestions(
                                tema, primeiro_nome, selected, original_kws
                            )

                            st.session_state.suggested_strings = gemini.generate_search_strings(
                                tema, 
                                selected, 
                                original_kws,
                                st.session_state.suggested_keywords
                            )

                            st.session_state.interpretation_generated = True
                            if 'id_usuario' in st.session_state:
                                atualizar_termos_sugeridos(
                                    st.session_state.id_usuario,
                                    st.session_state.suggested_keywords
                                )

                        st.session_state.sub_step = 'c'
                        st.rerun()
                else:
                    st.button("Gerar Relatório de Delineamento ▶️", disabled=True, width="stretch", key="btn_gerar_disabled")

            rodape_institucional()

        # ========== SUB-ETAPA 2c: RELATÓRIO ==========
        elif sub_step == 'c':
            selected = st.session_state.get('selected_concepts', [])

            col_nav1, col_nav2 = st.columns([1, 3])
            with col_nav1:
                if st.button("⬅️ Voltar à Seleção", key="btn_voltar_2c"):
                    st.session_state.sub_step = 'b'
                    st.rerun()

            render_etapa_2c(d, r, selected)

            st.subheader("🔎 Chaves de Busca Sugeridas")
            st.caption("Copie as chaves de busca abaixo para usar no Painel ou em bases de dados")

            suggested_strings = st.session_state.get('suggested_strings', {})

            if suggested_strings:
                for key, data in suggested_strings.items():
                    with st.container(border=True, key=f"string_container_{key}"):
                        st.markdown(f"**{data.get('titulo', key)}**")
                        st.caption(data.get('descricao', ''))
                        col_str, col_btn = st.columns([4, 1])
                        with col_str:
                            st.code(data.get('string', ''), language='text')
                        with col_btn:
                            if st.button("📋 Copiar", key=f"copy_{key}", width="stretch"):
                                st.session_state.dashboard_query = data.get('string', '')
                                st.session_state.dashboard_query_source = "delineascópio"
                                st.toast("✅ Chave copiada!")
            else:
                search_string = r.get('search_string', 'N/A')
                with st.container(border=True):
                    st.markdown("**🔎 Chave de Busca Original**")
                    col_str, col_btn = st.columns([4, 1])
                    with col_str:
                        st.code(search_string, language='text')
                    with col_btn:
                        if st.button("📋 Copiar", key="copy_original", width="stretch"):
                            st.session_state.dashboard_query = search_string
                            st.session_state.dashboard_query_source = "delineascópio"
                            st.toast("✅ Chave copiada!")

            st.subheader("🔬 Transparência: Chave de Busca Usada")
            st.caption("Esta é a chave de busca exata que foi usada para recuperar artigos do OpenAlex")
            
            with st.container(border=True):
                search_objective = r.get('search_objective', '')
                if search_objective:
                    st.markdown(f"**Objetivo:** {search_objective}")
                    st.divider()
                
                search_string = r.get('search_string', 'N/A')
                st.markdown("**Chave de busca executada:**")
                col_code, col_copy = st.columns([4, 1])
                with col_code:
                    st.code(search_string, language='text')
                with col_copy:
                    if st.button("📋 Copiar", key="copy_search_string_2c", width="stretch"):
                        st.session_state.dashboard_query = search_string
                        st.session_state.dashboard_query_source = "delineascópio"
                        st.toast("✅ Chave copiada para o Painel!")
                
                articles_count = r.get('articles_count', 0)
                graph_stats = r.get('graph_stats', {})
                st.caption(f"📊 Resultados: {articles_count} artigos encontrados | "
                          f"{graph_stats.get('nodes', 0)} conceitos | "
                          f"{graph_stats.get('edges', 0)} coocorrências")

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                try:
                    cache_key = f"pdf_{d.get('nome', '')}_{d.get('timestamp', '')}"
                    if st.session_state.get('pdf_cache_key') != cache_key:
                        st.session_state.cached_pdf_bytes = generate_pdf_report(
                            form_data=d,
                            result=r,
                            selected_concepts=selected,
                            suggested_keywords=st.session_state.get('suggested_keywords', []),
                            suggested_strings=st.session_state.get('suggested_strings', {}),
                            badges=st.session_state.get('badges', [])
                        )
                        st.session_state.pdf_cache_key = cache_key
                    
                    st.download_button(
                        "📥 Baixar PDF Completo",
                        data=st.session_state.cached_pdf_bytes,
                        file_name=f"delineamento_{d['nome'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        width='stretch',
                        type="primary",
                        key="dl_pdf_relatorio"
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {str(e)}")

            with col2:
                if st.button("📝 Avaliar Sistema", type="primary", width="stretch", key="btn_avaliar_2c"):
                    st.session_state.step = 3
                    st.rerun()

            st.info("""
            🎉 **Parabéns!** Você completou a trilha de delineamento!

            Agora você pode:
            - 📥 **Baixar o PDF** com o relatório completo
            - 🔬 **Usar a Interação** para explorar o grafo
            - 📋 **Realizar novos delineamentos**
            - 📜 **Usar o Histórico** para comparar diferentes delineamentos            
            - 🔎 **Usar o Painel** para analisar a literatura
            - 📝 **Avaliar o sistema** e nos ajudar a melhorar
            """)

            if st.button("🔄 Iniciar Novo Delineamento", width="stretch", key="btn_novo_2c"):
                st.session_state.step = 1
                st.session_state.resultado = None
                st.session_state.form_data = {}
                st.session_state.avaliacao_completa = False
                st.session_state.badges = []
                st.session_state.selected_concepts = []
                st.session_state.interpretation_generated = False
                st.session_state.personalized_interpretation = None
                st.session_state.suggested_keywords = []
                st.session_state.suggested_strings = {}
                st.rerun()

            rodape_institucional()

    # ========== ETAPA 3: AVALIAÇÃO EXPANDIDA ==========
    elif st.session_state.step == 3:
        st.header("⭐ 5. Avaliação")
        st.caption("Suas respostas são fundamentais para aprimorarmos o sistema!")

        st.info("""
📊 **Termo de Consentimento Livre e Esclarecido**
 
Convidamos você a participar da pesquisa sobre o uso de palavras-chave na pesquisa acadêmica. Sua participação é totalmente voluntária, e você pode desistir a qualquer momento sem nenhum prejuízo.

O objetivo do estudo é investigar como a avaliação automatizada de definições preliminares de um projeto, como tema, questão de pesquisa e palavras-chave, pode apoiar estudantes no delineamento do escopo do estudo e na delimitação mais precisa de suas propostas.

Ressaltamos que nenhuma informação identificável é utilizada na pesquisa.

Caso tenha dúvidas ou necessite de mais informações, entre em contato por e-mail com o pesquisador responsável, Rafael Antunes dos Santos (rafael.antunes@ufrgs.br ou rderafa@gmail.com), doutorando do Programa de Pós-Graduação em Informática na Educação, da Universidade Federal do Rio Grande do Sul.
                
Para prosseguir com o preenchimento deste formulário, assinale a alternativa mais conveniente à sua decisão. Ao assinalar que concorda, você declara que entende o objetivo da pesquisa e concorda em participar voluntariamente.
""")

        # Botão para download do TCLE completo
        with open("assets/TCLE_Delineia.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Baixar TCLE Completo (PDF)",
                data=pdf_file,
                file_name="TCLE_Delineia.pdf",
                mime="application/pdf",
                help="Clique para baixar o Termo de Consentimento Livre e Esclarecido completo",
                key="dl_tcle_pdf"
            )

        st.markdown("") # Um pequeno espaço
        tcle_aceite = st.checkbox(
            "📝 Li, compreendi e **CONCORDO** em participar da Etapa 1 (formulários online).",
            key="tcle_aceite"
        )

        st.markdown("") # Um pequeno espaço
        tcle_rejeita = st.checkbox(
            "📝 Li, mas **NÃO CONCORDO** em participar desta pesquisa.",
            key="tcle_rejeita"
        )

        # Validação de exclusão mútua do TCLE
        if tcle_aceite and tcle_rejeita:
            st.warning("⚠️ Por favor, selecione apenas uma opção: CONCORDO ou NÃO CONCORDO.")
        elif tcle_aceite:
            st.success("✅ Obrigado por concordar em participar!")
        elif tcle_rejeita:
            st.info("📋 Entendido. Você ainda pode explorar o sistema, mas suas respostas não serão coletadas.")

        with st.form("formulario_avaliacao"):

            # ==================== SEÇÃO 1: UTILIDADE PERCEBIDA ====================
            st.subheader("💼 Utilidade Percebida")

            q1 = st.radio(
                "F2.1. Usar o Delinéia melhora a minha capacidade de escolha de palavras-chave para o escopo da pesquisa",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q1"
            )

            q2 = st.radio(
                "F2.2. Usar o Delinéia aumenta minha produtividade na definição do projeto",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q2"
            )

            q3 = st.radio(
                "F2.3. O Delinéia é útil para delimitar meu projeto de pesquisa",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q3"
            )

            q4 = st.radio(
                "F2.4. O Delinéia me ajuda a posicionar meu projeto na literatura do meu tema",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q4"
            )

            st.divider()

            # ==================== SEÇÃO 2: FACILIDADE DE USO ====================
            st.subheader("🎯 Facilidade de Uso Percebida")

            q5 = st.radio(
                "F2.5. O Delinéia é fácil de usar",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q5"
            )

            q6 = st.radio(
                "F2.6. A interação com o Delinéia é clara e compreensível",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q6"
            )

            q7 = st.radio(
                "F2.7. A navegação entre as diferentes funcionalidades é intuitiva",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q7"
            )

            st.divider()

            # ==================== SEÇÃO 3: QUALIDADE DA INFORMAÇÃO ====================
            st.subheader("📊 Qualidade da Informação")

            q8 = st.radio(
                "F2.8. As análises e sugestões do Delinéia são relevantes para meu projeto",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q8"
            )

            q9 = st.radio(
                "F2.9. A avaliação gerada pela IA é construtiva para meu projeto",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q9"
            )

            q10 = st.radio(
                "F2.10. As chaves de busca que foram oferecidas são precisas para o meu tema",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q10"
            )

            q11 = st.radio(
                "F2.11. O grafo de coocorrências me ajudou a visualizar relações entre conceitos",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q11"
            )

            q12 = st.radio(
                "F2.12. O Delinéia me ajudou a formular perguntas de pesquisa mais precisas",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q12"
            )

            q13 = st.radio(
                "F2.13. O relatório em PDF é adequado para apresentar ao meu orientador",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q13"
            )

            st.divider()

            # ==================== SEÇÃO 4: INTENÇÃO DE USO ====================
            st.subheader("🔮 Intenção de Uso")

            q14 = st.radio(
                "F2.14. O tempo gasto usando o Delinéia compensa os resultados obtidos",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q14"
            )

            q15 = st.radio(
                "F2.15. Eu pretendo usar o Delinéia em projetos futuros",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q15"
            )

            q16 = st.radio(
                "F2.16. Eu usaria o Delinéia em diferentes fases da minha pesquisa (projeto, qualificação, defesa)",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q16"
            )

            st.divider()

            # ==================== SEÇÃO 5: CONFIANÇA NO SISTEMA ====================
            st.subheader("🔒 Confiança no Sistema")

            q17 = st.radio(
                "F2.17. Eu confio nas análises geradas pelo Delinéia",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q17"
            )

            q18 = st.radio(
                "F2.18. Eu me sinto confortável em basear decisões acadêmicas com os resultados do Delinéia",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q18"
            )

            st.divider()

            # ==================== SEÇÃO 6: EXPERIÊNCIA DO USUÁRIO ====================
            st.subheader("✨ Experiência do Usuário")

            q19 = st.radio(
                "F2.19. O design da interface é agradável",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q19"
            )

            q20 = st.radio(
                "F2.20. O tempo de processamento do relatório foi adequado",
                ["Concordo Totalmente", "Concordo", "Neutro", "Discordo", "Discordo Totalmente"],
                horizontal=True,
                key="q20"
            )

            st.divider()

            # ==================== SEÇÃO 7: NET PROMOTER SCORE ====================
            st.subheader("⭐ Satisfação Geral (Net Promoter Score)")

            nps = st.slider(
                "F2.21. Em uma escala de 0 a 10, quanto você recomendaria o Delinéia para um colega?",
                min_value=0,
                max_value=10,
                value=7,
                help="0 = Definitivamente não recomendaria | 10 = Definitivamente recomendaria"
            )

            # Mostrar categoria NPS em tempo real
            if nps >= 9:
                st.success("🌟 **Promotor** - Obrigado pelo entusiasmo!")
            elif nps >= 7:
                st.info("😐 **Neutro** - O que podemos melhorar?")
            else:
                st.warning("😞 **Desanimado** - Queremos ouvir suas sugestões!")

            st.divider()

            # ==================== SEÇÃO 8: COMENTÁRIOS ADICIONAIS ====================
            st.subheader("💬 Comentários Adicionais")

            q22 = st.text_area(
                "F2.22. O que você mais gostou no Delinéia?",
                height=100,
                key="q22",
                placeholder="Descreva os aspectos mais positivos da sua experiência..."
            )

            q23 = st.text_area(
                "F2.23. O que poderia ser melhorado?",
                height=100,
                key="q23",
                placeholder="Sugestões de melhorias, funcionalidades ausentes, problemas encontrados..."
            )

            q24 = st.text_area(
                "F2.24. Funcionalidades que você gostaria de ver no futuro:",
                height=100,
                key="q24",
                placeholder="Ideias para próximas versões..."
            )

            q25 = st.text_area(
                "F2.25. Como você usou (ou pretende usar) os resultados do Delinéia na sua pesquisa?",
                height=100,
                key="q25",
                placeholder="Ex: projeto de qualificação, artigo, revisão de literatura..."
            )

            st.divider()

            # ==================== SEÇÃO 9: AUTOAVALIAÇÃO ====================
            st.subheader("🔄 Autoavaliação")

            st.markdown("""
            **Reflexão sobre seu processo:**  
            No formulário inicial (F1.5), você indicou seu nível de segurança em relação às palavras-chave escolhidas.  
            Agora, após ter lido o relatório e as análises do Delinéia, como você avalia sua escolha inicial?
            """)

            q26 = st.radio(
                "F2.26. Considerando as palavras-chave escolhidas inicialmente e a leitura do relatório, qual seu nível de segurança em relação às palavras-chave que você definiu para a pesquisa bibliográfica do seu projeto?",
                ["Totalmente seguro", "Seguro", "Neutro", "Inseguro", "Totalmente inseguro"],
                horizontal=True,
                key="q26"
            )

            # Mostrar comparação se disponível
            if 'form_data' in st.session_state and 'confianca' in st.session_state.form_data:
                confianca_inicial = st.session_state.form_data['confianca']
                st.info(f"💡 **Sua resposta inicial (F1.5):** {confianca_inicial}")

            st.divider()

            # ==================== SEÇÃO 10: PERFIL DO RESPONDENTE ====================
            st.subheader("👤 Perfil do Respondente (Opcional)")

            col1, col2 = st.columns(2)

            with col1:
                q27 = st.selectbox(
                    "F2.27. Nível acadêmico:",
                    ["Prefiro não informar", "Graduação", "Especialização", "Mestrado",
                     "Doutorado", "Pós-Doutorado", "Docente"],
                    key="q27"
                )

                q28 = st.selectbox(
                    "F2.28. Experiência prévia com bibliometria:",
                    ["Nenhuma", "Básica", "Intermediária", "Avançada"],
                    key="q28"
                )

            with col2:
                q29 = st.selectbox(
                    "F2.29. Área do conhecimento:",
                    ["Prefiro não informar", "Ciências Exatas", "Ciências Biológicas", "Ciências da Saúde",
                     "Ciências Agrárias", "Ciências Sociais Aplicadas", "Ciências Humanas",
                     "Linguística/Letras/Artes", "Engenharias", "Multidisciplinar"],
                    key="q29"
                )

                q30 = st.selectbox(
                    "F2.30. Tempo gasto usando o Delinéia hoje:",
                    ["< 15 min", "15-30 min", "30-60 min", "> 1 hora"],
                    key="q30"
                )

            st.divider()

            # ==================== SEÇÃO 11: CONVITE À CONTINUIDADE ====================
            st.subheader("🤝 Convite à Continuidade da Pesquisa")

            st.markdown("""
            **Queremos continuar contando com você!**
            
            Esta pesquisa não termina aqui. Estamos desenvolvendo novas funcionalidades e gostaríamos 
            de convidá-lo(a) para participar de outras etapas do estudo, como:
            
            - 🎥 **Sessões mediadas por videoconferência** para observação de uso
            - 🎓 **Oficinas e treinamentos** sobre bibliometria e ferramentas de pesquisa
            - 🧪 **Testes de novas funcionalidades** antes do lançamento público
            - 📊 **Entrevistas em profundidade** sobre suas estratégias de pesquisa
            
            Sua participação é voluntária e você poderá desistir a qualquer momento. 
            Caso aceite, entraremos em contato por e-mail com mais informações.
            """)

            aceite_continuidade = st.checkbox(
                "✅ **CONCORDO** em ser convidado(a) para atividades com gravação de áudio e vídeo.",
                key="aceite_continuidade",
                help="Ao marcar esta opção, você demonstra interesse em contribuir com o desenvolvimento do Delinéia"
            )

            rejeita_continuidade = st.checkbox(
                "✅ **NÃO CONCORDO** em participar de atividades qualitativas com gravação.",
                key="rejeita_continuidade",
                help="Você não será considerado em convites de continuidade da pesquisa."
            )

            if aceite_continuidade and not rejeita_continuidade:
                st.success("🎉 Obrigado por aceitar continuar conosco! Você receberá um e-mail com mais informações em breve.")
            elif rejeita_continuidade and not aceite_continuidade:
                st.info("🚫 Você não será considerado em convites de continuidade da pesquisa.")
            elif aceite_continuidade and rejeita_continuidade:
                st.warning("⚠️ Por favor, selecione apenas uma opção: CONCORDO ou NÃO CONCORDO.")

            st.divider()

            # ==================== BOTÃO DE ENVIO ====================
            submitted = st.form_submit_button(
                "📤 Enviar Avaliação",
                type="primary",
                width="stretch"
            )

            if submitted:
                # Validação obrigatória dos checkboxes do TCLE
                tcle_valido = True
                
                # Validação 1: Concordância inicial (deve marcar exatamente uma opção)
                if tcle_aceite and tcle_rejeita:
                    st.error("⚠️ **TCLE - Concordância:** Selecione apenas UMA opção (CONCORDO ou NÃO CONCORDO).")
                    tcle_valido = False
                elif not tcle_aceite and not tcle_rejeita:
                    st.error("⚠️ **TCLE - Concordância:** É obrigatório selecionar uma opção (CONCORDO ou NÃO CONCORDO).")
                    tcle_valido = False
                
                # Validação 2: Continuidade (deve marcar exatamente uma opção)
                if aceite_continuidade and rejeita_continuidade:
                    st.error("⚠️ **Convite à Continuidade:** Selecione apenas UMA opção (CONCORDO ou NÃO CONCORDO).")
                    tcle_valido = False
                elif not aceite_continuidade and not rejeita_continuidade:
                    st.error("⚠️ **Convite à Continuidade:** É obrigatório selecionar uma opção (CONCORDO ou NÃO CONCORDO).")
                    tcle_valido = False
                
                # Impede envio se validação falhar
                if not tcle_valido:
                    st.warning("📋 Por favor, revise suas escolhas no TCLE e no Convite à Continuidade antes de enviar.")
                    st.stop()
                
                # Calcular categoria NPS
                if nps >= 9:
                    nps_category = "Promotor 🌟"
                elif nps >= 7:
                    nps_category = "Neutro 😐"
                else:
                    nps_category = "Detrator 😞"

                # Armazenar respostas
                avaliacao_data = {
                    # Perguntas Likert (F2.1-F2.20)
                    'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5,
                    'q6': q6, 'q7': q7, 'q8': q8, 'q9': q9, 'q10': q10,
                    'q11': q11, 'q12': q12, 'q13': q13, 'q14': q14, 'q15': q15,
                    'q16': q16, 'q17': q17, 'q18': q18, 'q19': q19, 'q20': q20,
                    # NPS (F2.21)
                    'nps': nps,
                    'nps_category': nps_category,
                    # Campos abertos (F2.22-F2.25)
                    'q22': q22,
                    'q23': q23,
                    'q24': q24,
                    'q25': q25,
                    # Autoavaliação (F2.26)
                    'q26': q26,
                    # Perfil (F2.27-F2.30)
                    'q27': q27,
                    'q28': q28,
                    'q29': q29,
                    'q30': q30,
                    # Convite à continuidade
                    'tcle_aceite': tcle_aceite,
                    'tecle_rejeita': tcle_rejeita,
                    'aceite_continuidade': aceite_continuidade,
                    'rejeita_continuidade': rejeita_continuidade,
                    # Metadados
                    'timestamp': datetime.now().isoformat()
                }

                # Salvar em session_state
                st.session_state.avaliacao_completa = True
                st.session_state.avaliacao_data = avaliacao_data

                # Enviar para Google Sheets
                if 'id_usuario' in st.session_state:
                    enviar_formulario_avaliacao(
                        st.session_state.id_usuario,
                        avaliacao_data
                    )

                # Badge de conclusão (Agora usando a função g() para o gênero correto)
                badge_final = f'💎 {g("Avaliador", "Avaliadora")}'
                add_badge(badge_final)

                # Feedback visual
                st.session_state.mostrar_resumo_final = True
                st.success("✅ Avaliação enviada com sucesso!")
                
        if st.session_state.get('mostrar_resumo_final'):
            
            # Resumo da avaliação
            dados = st.session_state.get('avaliacao_data', {})
            rec_nps = dados.get('nps', 0)
            rec_cat = dados.get('nps_category', '-')
            rec_q27 = dados.get('q27', '-')
            rec_q28 = dados.get('q28', '-')
            rec_q29 = dados.get('q29', '-')
            rec_q30 = dados.get('q30', '-')
            
            aceite = dados.get('aceite_continuidade', False)
            msg_continuidade = "Sim ✅" if aceite else "Não"
               
            st.info(f"""
            📊 **Resumo da sua avaliação:**

            - **NPS:** {rec_nps}/10 ({rec_cat})
            - **Nível acadêmico:** {rec_q27}
            - **Experiência bibliométrica:** {rec_q28}
            - **Área:** {rec_q29}
            - **Tempo de uso:** {rec_q30}
            - **Aceite para continuidade:** {msg_continuidade}

            🏆 **Badge desbloqueado:** {g("Delineador", "Delineadora")}

            Obrigado por dedicar seu tempo para avaliar o Delinéia!
            Sua avaliação é essencial para o desenvolvimento contínuo do sistema.

            **Clique no botão abaixo para concluir e visualizar suas conquistas.**
            """)

            # BOTÕES DE AÇÃO
            col_pdf_aval, col_resgatar = st.columns([1, 1])
            
            with col_pdf_aval:
                try:
                    from pdf_generator import generate_evaluation_pdf
                    
                    if 'cache_pdf_avaliacao' not in st.session_state:
                        st.session_state.cache_pdf_avaliacao = generate_evaluation_pdf(
                            form_data=st.session_state.get('form_data', {}),
                            avaliacao_data=st.session_state.get('avaliacao_data', {})
                        )
                    
                    nome_aluno = st.session_state.get('form_data', {}).get('nome', 'aluno').split()[0]
                    nome_arquivo = f"avaliacao_{nome_aluno}.pdf"
                    
                    st.download_button(
                        label="📥 Salvar Avaliação (PDF)",
                        data=st.session_state.cache_pdf_avaliacao,
                        file_name=nome_arquivo,
                        mime="application/pdf",
                        width="stretch",
                        key="dl_avaliacao_pdf"
                    )
                except Exception as e:
                    st.warning(f"PDF indisponível: {e}")
            
            with col_resgatar:
                if st.button("🏆 Resgatar Conquistas", type="primary", width="stretch"):
                    st.session_state.step = 4
                    st.session_state.mostrar_resumo_final = False
                    st.rerun()
                    
        rodape_institucional()
    
    # ========== ETAPA 4: CONCLUSÃO ==========
    if st.session_state.step == 4:
        st.success("🎉 Parabéns! Você completou todas as etapas!")
        st.markdown(f"### 🏆 Conquista Desbloqueada: {g('Delineador', 'Delineadora')}!")

        primeiro_nome = st.session_state.form_data['nome'].split()[0]

        st.write(f"**{primeiro_nome}**, você concluiu com sucesso:")
        st.write("✅ Delineamento completo do projeto")
        st.write("✅ Análise bibliométrica avançada")
        st.write("✅ Avaliação do sistema Delinéia")
        st.write(f"\n**🏅 Suas conquistas:** {' '.join(st.session_state.badges)}")

        st.divider()

        # ========== PRÊMIO: VÍDEO MUSICAL ==========
        st.markdown("### 🎵 Prêmio Especial: Uma palavra no escuro")
        
        st.markdown("""
        <div style="text-align: justify; 
                    background-color: #ffffff; 
                    border-left: 4px solid #28a745; 
                    padding: 1rem; 
                    border-radius: 0.25rem;
                    color: #000000;">
        Como reconhecimento pela sua dedicação, presenteamos você com uma obra que simboliza 
        o processo de construção do conhecimento: a busca por palavras que iluminam 
        caminhos no escuro da incerteza. Uma homenagem à Jorge Luis Borges e à sua Biblioteca de Babel.
        <div>
        """, unsafe_allow_html=True)

        # Embedar vídeo do YouTube
        video_url = "https://www.youtube.com/embed/aoKVEJc-7MU"
        
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin: 2rem 0;">
                <iframe width="700" height="394" 
                        src="{video_url}" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Créditos em expander
        with st.expander("📜 Créditos e Informações", expanded=True):
            st.markdown("""
            <div style="text-align: center; 
                        background-color: #f8f9fa; 
                        padding: 1.5rem; 
                        border-radius: 0.5rem;
                        color: #000000;">
            
            **Título:** A palavra no escuro ou os dialetos do poço
                        
            **Álbum:** Os olhos de Borges (Versão musical do livro homônimo)
                        
            **Livro:** BRASIL, J.V. *Os olhos de Borges*. Porto Alegre: WS Editor, 1997.
                                            
            **Intérprete(s):** Hique Gomez

            **Letra:** Jaime Vaz Brasil
                                    
            **Música:** Hique Gomez 
                                   
            **Produção:** FUMPROARTE/POA e Instituto Fernando Pessoa
                                    
            **Ano:** 1999
            
            ---
            
            **Conexão com o Delinéia:**
            
            Esta música integra o universo poético que inspira a construção do sistema Delinéia. 
            A metáfora da "palavra no escuro" ecoa o processo de delineamento do escopo de pesquisa: 
            buscar, na vastidão da literatura científica, as palavras-chave que iluminam o caminho 
            do conhecimento.
            
            Assim como os "dialetos do poço" sugerem múltiplas vozes emergindo da profundidade, 
            o Delinéia revela as múltiplas dimensões conceituais que estruturam um campo de pesquisa, 
            auxiliando estudantes a encontrarem suas próprias vozes acadêmicas.
            </div>
            """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("📜 Leia o prólogo da tese", width="stretch"):
                st.session_state.open_prologo = True
                st.info("""
                **O Delineascópio**

Esta é uma palavra que respira. *Delineamento*…

Não é uma palavra-ponto, uma palavra-fim. Não é limite, fronteira ou conclusão. Embora o Novo Dicionário Aurélio, em sua precisão cartográfica, nos diga que delinear tenha o significado de "[...] fixar os limites de; estremar, demarcar", a verdadeira alma da palavra reside em sua outra definição: "[...] traçar as linhas gerais de; esboçar, debuxar".[^1] Esta é uma palavra-processo. Uma palavra-verbo que se disfarça de substantivo. No seu coração, pulsa o ato de delinear, do latim *delineare*, "[...] por via erudita".[^2] Em sua família, registrada nas colunas dos léxicos do vocabulário ortográfico da Academia Brasileira de Letras[^3], encontramos o delineador (aquele que traça) e o delineado (aquilo que foi traçado). Mas o delineamento é algo mais. É o "[...] ato de delinear".[^4] Não é o traço, mas o traçar. Não é o mapa, mas o mapear. É "[...] o primeiro esboço ou projeto de qualquer obra; plano geral".[^5] É o gesto inaugural da criação. É o primeiro traço.

O Dicionário Houaiss nos conta um segredo: a palavra já circulava em 1552[^6]. Mil quinhentos e cinquenta e dois. Pensemos nisso. Esta não é uma palavra da Revolução Industrial. Não nasceu fria, metálica, otimizada sob uma linha de montagem, no distanciamento entre idealização e produção. Ela nasceu no auge do Renascimento, numa era de exploração febril, quando o mundo conhecido se expandia e exigia ser desenhado, quando os mapas eram mais suposição do que certeza. Sua primeira definição registrada, "[...] ato ou efeito de delinear(-se); esboço, delineação, traçado [...]", era usada para o "[...] esboço do projeto de reparação a ser feita em qualquer parte de embarcação".[^7]

Que poética inaugural! O delineamento não é sobre construir o navio do zero; é sobre o reparo. É sobre olhar para uma estrutura que já existe (uma ideia, um navio, um campo de estudo) e traçar o plano para sua renovação, sua travessia. O navio da pesquisa do estudante já existe, ancorado no porto da sua intuição, mas com o casco opaco de incertezas. Ele precisa ser delineado para poder navegar.

Frequentemente, quando buscamos uma tradução apressada, a língua inglesa nos oferece, friamente, *design*. Mas *design* é uma palavra que trai a alma do delineamento. *Design* carrega o peso da indústria, do produto final, da ergonomia. O *design* é assertivo, muitas vezes dogmático. Ele se impõe à matéria. O *design* é a cadeira, o *iPhone*, o motor: soluções acabadas, polidas, fechadas em si. Delineamento é o oposto. É uma palavra de escuta. O delineamento é a pergunta ganhando forma.

O verbo delinear, "[...] traçar as linhas gerais de; esboçar, debuxar",[^8] é um ato de humildade. O delineador não inventa o contorno do continente; ele traça o contorno que descobre. O *design* fecha; o delineamento abre. O *design* é a certeza do engenheiro; o delineamento é a hesitação do artista diante da tela em branco. É por isso que o delineamento é a palavra-raiz da cultura das descobertas, sejam elas artísticas, filosóficas, científicas ou mesmo industriais. A descoberta não é um *design*, mas um delineamento. É o ato de tatear no escuro e, aos poucos, "[...] traçar as linhas gerais, o plano de; projetar, planejar".[^9] É a transformação da incerteza em foco.

E aqui, uma busca interessante se revela. A palavra delineamento é, em si, um ato da crítica, da análise e da academia, mas é rara dentro da prosa de ficção ou dos versos de poesia. Parece ser uma palavra que usamos para observar a literatura, e não uma palavra que a literatura usa para observar o mundo. Um romancista provavelmente escreveria "o contorno do seu rosto" ou "o traçado do plano", mas raramente "o delineamento do seu rosto". A palavra pertence ao analista, ao pesquisador. A encontramos em textos de crítica literária, operando do mesmo modo como esta tese propõe: o processo de dar forma, traçar perfis e estruturar a descoberta.

Um ensaio sobre Erico Verissimo e Graham Greene menciona o "[...] delineamento de dois perfis de personagens [...]".[^10] Um estudo sobre Machado de Assis foca no "[...] delineamento do percurso da reelaboração de 'O alienista' [...]",[^11] analisando como o escritor esboçou e refez sua obra. Um crítico, sobre o poeta Demétrio Vieira Diniz, afirma que seu livro "[...] atesta e faz saber o delineamento de uma singular dicção".[^12] É a palavra que usamos para entender a criação, perfeita para descrever o processo de descoberta que o aluno, no centro desta tese, está colocado a realizar.

Em um canto esquecido da estante, em um Dicionário de Comunicação, encontramos um artefato. Ao buscar uma remissiva da entrada principal do verbete "Episcópio", lemos: "Aparelho baseado na reflexão de luz, que se destina à projeção de imagens de objetos opacos (tais como fotografias, desenhos etc.). Também chamado de delineascópio ou de projetor opaco".[^13] *Delineascópio*: o-que-projeta-o-delineado [sic].

Um aparelho (*scópio*) que torna visível (*projeção*) um esboço ou traçado (*delínea*). Aqui, a poética se completa. O conhecimento começa não com a luz, mas com um objeto opaco. Qual é o "objeto opaco" senão a ideia inicial de um pesquisador? É aqui que a palavra encontra sua casa nesta tese: "*Grandes modelos de linguagem e análise de coocorrência de palavras-chave para o delineamento do escopo de projetos de pesquisa no ensino superior*".

O estudante chega ao ensino superior carregando esse objeto opaco. Ele o segura nas mãos. O projeto o chama, em linguagem técnica, de "necessidade de informação", de uma etapa de "formulação" marcada por "sentimentos iniciais de dúvida e confusão", ou o estado de "pré-foco" onde a "incerteza é um estado cognitivo que comumente causa sintomas afetivos de ansiedade e falta de confiança".[^14] Como encontrar as agulhas certas nos palheiros mais loucos?

Simbolicamente, é uma intuição turva. Um vulto. Um interesse que ainda não tem palavras. É um desenho que não pode ser visto. É um mapa por fazer. Como traçar o que ainda não se vê? É preciso, então, um delineascópio. Um aparelho de luz refletida. Não a luz que cega, mas a que projeta os contornos do que já está lá. Esta tese é uma das engrenagens da engenharia desta máquina. O estudante coloca seu objeto opaco (sua ideia de tema, sua questão de pesquisa inicial, suas primeiras palavras-chave) na máquina. A máquina, então, usa duas fontes de luz para projetar essa ideia na grande teia da literatura científica.

A primeira luz é a análise de coocorrência de palavras. Ela funciona exatamente como um episcópio: ela reflete a luz sobre o objeto opaco do aluno e projeta as conexões que ele não podia ver. O estudante vê seu termo (por exemplo, "*gamification*") e, de repente, projetado na tela, ele o vê ligado a "*motivation*", "*higher education*", "*engagement*", "*learning outcomes*". O grafo de coocorrência é a projeção. O opaco tornou-se visível, relacional, delineável. O estudante pode, agora, pegar seu lápis e traçar as conexões que a luz revelou. A máquina oferece uma visão complementar dos conceitos centrais.

A segunda luz é generativa. São os grandes modelos de linguagem (LLMs). Se a coocorrência é a projeção, o LLM é o *feedback*, a mediação. É a voz que ajuda o estudante a ajustar o foco do delineascópio. Ele não se limita a projetar o que existe; ele conversa com a projeção. Ele oferece o *feedback* textual automatizado. Ele olha para a projeção e sussurra: "As palavras-chave designadas para o projeto se mostram alinhadas... No entanto, algumas expressões ainda podem ser consideradas genéricas... É recomendável que você considere a possibilidade de incorporar termos mais descritivos… Converse com seu orientador…".[^15] Ele sugere novas lentes, novas palavras. O delineamento do escopo deixa de ser uma tarefa burocrática de definição de limites e se torna um ato poético de projeção e descoberta. Deixa de ser um ato de solidão e passa a ser um ato de mediação. E no centro deste ato, o estudante. Este projeto coloca o aluno no centro desse processo. O estudante não é um receptor passivo de *design*. Ele é o delineador[^16].

Esta tese reconhece a luta humana nesse processo. Ela se ancora em modelos teóricos que são, em essência, mapas da alma deste estudante-pesquisador. Ela se fundamenta no modelo de Kuhlthau, que entende a busca como uma passagem dolorosa e necessária da "incerteza" para a confiança.[^17] Ela se baseia no modelo cognitivo de escrita de Flower e Hayes, que entende a escrita não como uma tradução linear, mas como um processo recursivo de "planejamento, geração de ideias, organização e definição de metas"[^18], ou seja, o próprio ato de delinear. E se alicerça no modelo de comportamento informacional de Wilson, que mapeia o "comportamento de busca" e as "barreiras" que tornam a ideia opaca em primeiro lugar.[^19]

O delineamento proposto nesta tese é, portanto, terapêutico. Ele oferece ao estudante, que "enfrenta dificuldades" e "inúmeros desafios", as ferramentas não para resolver seu problema, mas para vê-lo projetado. A ferramenta torna-se uma mediadora do pensamento científico, um andaime para a autonomia investigativa, um fomento ao pensamento crítico.

Em 1552, o delineamento era o esboço para reparar um navio e prepará-lo para a travessia. Hoje, o delineamento é o esboço para reparar a confiança do estudante-pesquisador, dando-lhe o mapa: o delineascópio para sua própria travessia intelectual. A pesquisa, assim como a arte, não é sobre ter respostas prontas, mas sobre a coragem de fazer o traço inicial, de navegar a incerteza e, aos poucos, dar forma ao pensamento. Minha tese é a história da construção desse delineascópio. É um convite para trocar a ansiedade da página em branco pela descoberta mediada do primeiro traço. É uma palavra que acolhe a jornada do estudante, celebrando o esboço tanto quanto a obra final.

Para que todo estudante, segurando seu objeto opaco, possa encontrar a luz para projetá-lo e, enfim, começar a delinear…

… *Delinéia !!!*

---

**Notas:**

[^1]: FERREIRA, A.B.H. *Novo dicionário Aurélio da língua portuguesa*. 4.ed. Curitiba: Positivo, 2009.
[^2]: NASCENTES, A. *Dicionário etimológico resumido*. Rio de Janeiro: INL, 1966.
[^3]: ACADEMIA BRASILEIRA DE LETRAS. *Vocabulário ortográfico da língua portuguesa*. 5.ed. São Paulo: Global, 2009.
[^4]: FERREIRA, op. cit., p. 614.
[^5]: Ibid.
[^6]: HOUAISS, A.; VILLAR, M.S. *Dicionário Houaiss da língua portuguesa*. Rio de Janeiro: Objetiva, 2009.
[^7]: Ibid.
[^8]: FERREIRA, op. cit. p. 614.
[^9]: HOUAISS, op. cit., p. 610.
[^10]: DIAS, R.C. Americanos ingênuos e vietnamitas silenciosas: uma abordagem intertextual de O americano tranquilo e O prisioneiro. *Papéis*. Campo Grande, v. 23, n. 46, p. 61-75, 2019.
[^11]: CRESTANI, J.L. O Alienista: análise das variantes do folhetim e do livro. *SOLETRAS*, v. 10, n. 19, p. 156-166, 2010.
[^12]: DANTAS, M.L. O trem azul do destino da poesia de Demétrio Diniz. *Letras In.verso e Re.verso*. 2016.
[^13]: RABAÇA, C.A.; BARBOSA, G.G. *Dicionário de comunicação*. 2.ed. Rio de Janeiro: Campus, 2002.
[^14]: Trechos da tese sobre modelos de comportamento informacional.
[^15]: Exemplo de feedback gerado pelo sistema Delinéia.
[^16]: FERREIRA, op. cit., p. 614.
[^17]: KUHLTHAU, C.C. *Seeking meaning:* a process approach to library and information services. 2.ed. Westport: Libraries Unlimited, 2004.
[^18]: FLOWER, L.; HAYES, J.R. A cognitive process theory of writing. *College Composition and Communication*, v. 32, n. 4, p. 365–387, 1981.
[^19]: WILSON, T.D. On user studies and information needs. *Journal of Documentation*, v. 37, n. 1, p. 3-15, 1981.

**Desnorte**

O mundo é este monte: palha e pó.
 
Um caos de fibra, um tato quase cego,
 
Onde me perco e nada mais congrego,
 
Mergulhado em um vasto e mudo "só".

                        
Perdi o mapa; a rota é só tormento.
 
A perspectiva é turva, escura névoa;
 
A dúvida é um peso, noite, treva,
 
E o "quê fazer" corrói a cada momento.


A inércia abre a estrada do fracasso;
 
O não saber é um jugo, um precipício,
 
Não há repouso ou fim neste compasso.
                        

Resta encontrar, no caos, o puro indício:
 
A agulha. O aço. O ponto duro e escasso.
 
Que sangre o dedo, mas que estanque o vício.


🔍

                        
""")

        if st.button("🔄 Iniciar Novo Delineamento", width="stretch", key="btn_novo_etapa4"):
            st.session_state.step = 1
            st.session_state.resultado = None
            st.session_state.form_data = {}
            st.session_state.avaliacao_completa = False
            st.session_state.badges = []
            st.rerun()

            limpar_memoria()

        rodape_institucional()

# ==================== ABA 2: INTERAÇÃO (FUNÇÕES) ====================
with tab2:
    render_tab3_interacao()

# ==================== ABA 3: HISTÓRICO (Comparação e IA) ====================
with tab3:
    st.title("📜 Histórico e Comparação de Delineamentos")
    st.caption("Compare a evolução do seu escopo de pesquisa ao longo do tempo.")

    # Conectar ao Sheets
    sheet = conectar_google_sheets()
            
    if sheet:
        # 🔒 LÓGICA DE PRIVACIDADE E FILTRO DE USUÁRIO
        grafos_salvos = []
        user_id_atual = st.session_state.get('id_usuario')

        if not user_id_atual:
            # Se não tem usuário identificado, não mostra nada (Privacidade)
            st.warning("⚠️ Você precisa preencher o Formulário Inicial (Etapa 1) para acessar seu histórico privado.")
            # NÃO usar st.stop() aqui - ele para o script inteiro e impede a tab4 de renderizar
        else:
            # Se temos usuário logado/identificado, filtramos pelo ID dele
            grafos_salvos = exp.listar_grafos_salvos(sheet, user_id_atual)
                    
            if not grafos_salvos:
                st.info(f"Nenhum histórico encontrado para seu usuário atual. Salve um grafo na aba 'Exportação' primeiro.")

        # Se passou daqui, é porque tem grafos e é o usuário certo
        if grafos_salvos:
            st.subheader("1. Selecione os Delineamentos para Comparar")
            
            # 1. Cria a lista de opções com segurança antes de usar
            opcoes = [g['title'] for g in grafos_salvos]
            
            # Layout de seleção
            col_sel1, col_sel2 = st.columns(2)
            
            with col_sel1:
                # Lógica segura para o índice: se tiver lista, pega o penúltimo, senão 0
                idx_a = len(opcoes)-1 if len(opcoes) > 1 else 0
                g1_title = st.selectbox(
                    "Delineamento A (Referência/Antigo):", 
                    options=opcoes, 
                    index=idx_a,
                    key="sel_del_a" # Key única para evitar conflitos
                )
                
            with col_sel2:
                # Pega o primeiro (mais recente)
                g2_title = st.selectbox(
                    "Delineamento B (Atual/Recente):", 
                    options=opcoes, 
                    index=0,
                    key="sel_del_b" # Key única
                )

            # Botão de Ação
            if st.button("🔄 Comparar Delineamentos", type="primary", width="stretch", key="btn_comparar_hist"):
                if g1_title == g2_title:
                    st.warning("⚠️ Selecione dois delineamentos distintos para ver as diferenças.")
                else:
                    with st.spinner("⏳ Baixando dados e calculando similaridade..."):
                        ws1 = next(g['obj'] for g in grafos_salvos if g['title'] == g1_title)
                        ws2 = next(g['obj'] for g in grafos_salvos if g['title'] == g2_title)
                        
                        df1 = exp.carregar_grafo_do_sheets(ws1)
                        df2 = exp.carregar_grafo_do_sheets(ws2)
                        
                        if df1 is not None and df2 is not None:
                            st.session_state['df1_rico'] = df1
                            st.session_state['df2_rico'] = df2
                            st.session_state['comparacao_metrics'] = exp.calcular_comparacao(df1, df2)
                            st.session_state['comparacao_ativa'] = True
                            # Limpa análise anterior se houver
                            if 'ultima_analise_historico' in st.session_state:
                                del st.session_state['ultima_analise_historico']
                        else:
                            st.error("Erro ao ler os dados.")
                            st.session_state['comparacao_ativa'] = False

            # --- EXIBIR RESULTADOS (fora do if do botão) ---
            if st.session_state.get('comparacao_ativa', False) and 'comparacao_metrics' in st.session_state:
                metrics = st.session_state['comparacao_metrics']
                
                # RECUPERA METADADOS DOS NÓS (para usar em todas as listas)
                nodes_info = {}
                if 'df2_rico' in st.session_state and st.session_state['df2_rico'] is not None:
                    nodes_info = getattr(st.session_state['df2_rico'], 'attrs', {}).get('nodes_dict', {})
                if not nodes_info and 'df1_rico' in st.session_state and st.session_state['df1_rico'] is not None:
                    nodes_info = getattr(st.session_state['df1_rico'], 'attrs', {}).get('nodes_dict', {})
                
                st.divider()
                st.subheader("📊 Resultados da Comparação")
                
                # 1. Métricas Principais
                col_res1, col_res2, col_res3 = st.columns(3)
                
                col_res1.metric(
                    "Similaridade (Jaccard)", 
                    f"{metrics['jaccard']*100:.1f}%",
                    help="Mede o quanto os vocabulários se sobrepõem."
                )
                
                delta = metrics['qtd_2'] - metrics['qtd_1']
                col_res2.metric(
                    "Tamanho do Vocabulário", 
                    f"{metrics['qtd_2']} conceitos",
                    f"{delta:+}",
                    help="Diferença no número total de conceitos."
                )
                
                col_res3.metric(
                    "Novos Conceitos", 
                    len(metrics['exclusivos_novos']),
                    help="Conceitos que existem em B mas não em A."
                )
                
                # 2. Detalhamento Semântico (UM ABAIXO DO OUTRO)
                st.markdown("---")
                
                # LEGENDA DOS NÍVEIS (similar ao Mapa Temático)
                with st.expander("📖 Legenda: Níveis de Abstração (OpenAlex)", expanded=False):
                    st.markdown("""
                    O **OpenAlex** organiza conceitos científicos em 6 níveis hierárquicos de abstração (Level):
                    
                    - 🌍 **L0 - Raiz:** Grandes áreas do conhecimento (ex: Medicine, Science)
                    - 🙂 **L1 - Área:** Disciplinas amplas (ex: Biology, Psychology)
                    - 😊 **L2 - Campo:** Campos de estudo (ex: Genetics, Neuroscience)
                    - 🤔 **L3 - Subcampo:** Especializações (ex: Molecular biology)
                    - 🧐 **L4 - Tópico:** Tópicos específicos (ex: Gene expression)
                    - 🤓 **L5 - Específico:** Termos muito específicos (ex: CRISPR)
                    
                    **Interpretação:**  
                    Níveis baixos (L0-L2) = conceitos abrangentes  
                    Níveis altos (L4-L5) = conceitos específicos  
                    Conceitos são introduzidos nos mapas hierárquicos segundo a declaração de relevância presente em Score.
                    """)
                
                # === O QUE ENTROU (NOVIDADES) ===
                with st.container(border=True):
                    novos = metrics['exclusivos_novos']
                    st.markdown(f"#### 🆕 O que entrou ({len(novos)} novidades)")
                    
                    if novos:
                        tab_nov_map, tab_nov_list = st.tabs(["🗺️ Mapa Hierárquico", "🔤 Lista Alfabética"])
                        
                        with tab_nov_map:
                            # Classifica por level
                            novos_por_level = {i: [] for i in range(6)}
                            for c in novos:
                                if c in nodes_info:
                                    try:
                                        lvl = int(float(nodes_info[c].get('level', 5)))
                                        lvl = min(max(lvl, 0), 5)
                                        novos_por_level[lvl].append(c)
                                    except:
                                        novos_por_level[5].append(c)
                                else:
                                    novos_por_level[5].append(c)
                            
                            def top_by_score_nov(lista, n=5):
                                scored = [(c, nodes_info.get(c, {}).get('score', 0)) for c in lista]
                                scored.sort(key=lambda x: x[1], reverse=True)
                                return [c for c, _ in scored[:n]]
                            
                            def clean(s): return '"' + s.replace('"', "'").replace('\n', ' ') + '"'
                            
                            cores_nov = ["#dcfce7", "#bbf7d0", "#86efac", "#4ade80", "#22c55e", "#16a34a"]
                            font_nov = ["#14532d", "#14532d", "#14532d", "#14532d", "#ffffff", "#ffffff"]
                            labels = ["L0: Raiz", "L1: Área", "L2: Campo", "L3: Subcampo", "L4: Tópico", "L5: Específico"]
                            
                            graph_nov = '''digraph {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, margin="0.15,0.08"];
    nodesep=0.3; ranksep=0.6; bgcolor="transparent";
'''
                            total_nov = 0
                            niveis_nov = []
                            
                            for lvl in range(6):
                                top = top_by_score_nov(novos_por_level[lvl])
                                if top:
                                    niveis_nov.append(lvl)
                                    for c in top:
                                        label_n = f"{c}\\n({labels[lvl]})"
                                        graph_nov += f'    {clean(c)} [fillcolor="{cores_nov[lvl]}", fontcolor="{font_nov[lvl]}", label="{label_n}"];\n'
                                    graph_nov += f'    {{ rank=same; {" ".join([clean(c) for c in top])} }}\n'
                                    total_nov += len(top)
                            
                            for i in range(len(niveis_nov) - 1):
                                t1 = top_by_score_nov(novos_por_level[niveis_nov[i]], 1)
                                t2 = top_by_score_nov(novos_por_level[niveis_nov[i+1]], 1)
                                if t1 and t2:
                                    graph_nov += f'    {clean(t1[0])} -> {clean(t2[0])} [color="#86efac", style=dashed, arrowhead=none];\n'
                            
                            graph_nov += "}"
                            
                            try:
                                st.graphviz_chart(graph_nov, width="stretch")
                                st.caption(f"Top {total_nov} conceitos de {len(novos)} novidades, por relevância.")
                            except:
                                st.success(", ".join(sorted(novos)[:50]))
                        
                        with tab_nov_list:
                            conceitos_nov = sorted(novos)
                            num_cols = 4
                            tam_fatia = -(-len(conceitos_nov) // num_cols)
                            cols = st.columns(num_cols)
                            for i in range(num_cols):
                                with cols[i]:
                                    for c in conceitos_nov[i*tam_fatia:(i+1)*tam_fatia]:
                                        st.markdown(f"<div style='margin-bottom:2px; color:#16a34a;'>• {c}</div>", unsafe_allow_html=True)
                    else:
                        st.info("Nenhum conceito novo adicionado.")

                # === O QUE SAIU (REMOVIDOS) ===
                with st.container(border=True):
                    antigos = metrics['exclusivos_antigos']
                    st.markdown(f"#### 🗑️ O que saiu ({len(antigos)} removidos)")
                    
                    if antigos:
                        tab_ant_map, tab_ant_list = st.tabs(["🗺️ Mapa Hierárquico", "🔤 Lista Alfabética"])
                        
                        with tab_ant_map:
                            antigos_por_level = {i: [] for i in range(6)}
                            for c in antigos:
                                if c in nodes_info:
                                    try:
                                        lvl = int(float(nodes_info[c].get('level', 5)))
                                        lvl = min(max(lvl, 0), 5)
                                        antigos_por_level[lvl].append(c)
                                    except:
                                        antigos_por_level[5].append(c)
                                else:
                                    antigos_por_level[5].append(c)
                            
                            def top_by_score_ant(lista, n=5):
                                scored = [(c, nodes_info.get(c, {}).get('score', 0)) for c in lista]
                                scored.sort(key=lambda x: x[1], reverse=True)
                                return [c for c, _ in scored[:n]]
                            
                            cores_ant = ["#fee2e2", "#fecaca", "#fca5a5", "#f87171", "#ef4444", "#dc2626"]
                            font_ant = ["#7f1d1d", "#7f1d1d", "#7f1d1d", "#ffffff", "#ffffff", "#ffffff"]
                            
                            graph_ant = '''digraph {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, margin="0.15,0.08"];
    nodesep=0.3; ranksep=0.6; bgcolor="transparent";
'''
                            total_ant = 0
                            niveis_ant = []
                            
                            for lvl in range(6):
                                top = top_by_score_ant(antigos_por_level[lvl])
                                if top:
                                    niveis_ant.append(lvl)
                                    for c in top:
                                        label_a = f"{c}\\n({labels[lvl]})"
                                        graph_ant += f'    {clean(c)} [fillcolor="{cores_ant[lvl]}", fontcolor="{font_ant[lvl]}", label="{label_a}"];\n'
                                    graph_ant += f'    {{ rank=same; {" ".join([clean(c) for c in top])} }}\n'
                                    total_ant += len(top)
                            
                            for i in range(len(niveis_ant) - 1):
                                t1 = top_by_score_ant(antigos_por_level[niveis_ant[i]], 1)
                                t2 = top_by_score_ant(antigos_por_level[niveis_ant[i+1]], 1)
                                if t1 and t2:
                                    graph_ant += f'    {clean(t1[0])} -> {clean(t2[0])} [color="#fca5a5", style=dashed, arrowhead=none];\n'
                            
                            graph_ant += "}"
                            
                            try:
                                st.graphviz_chart(graph_ant, width="stretch")
                                st.caption(f"Top {total_ant} conceitos de {len(antigos)} removidos, por relevância.")
                            except:
                                st.error(", ".join(sorted(antigos)[:50]))
                        
                        with tab_ant_list:
                            conceitos_ant = sorted(antigos)
                            cols = st.columns(4)
                            tam = -(-len(conceitos_ant) // 4)
                            for i in range(4):
                                with cols[i]:
                                    for c in conceitos_ant[i*tam:(i+1)*tam]:
                                        st.markdown(f"<div style='margin-bottom:2px; color:#dc2626;'>• {c}</div>", unsafe_allow_html=True)
                    else:
                        st.info("Nenhum conceito foi removido.")
                
                # 3. NÚCLEO ESTÁVEL (TESAURO VISUAL HIERÁRQUICO)
                comuns = metrics['comuns']
                
                with st.container(border=True):
                    st.subheader(f"🌳 Núcleo Estável ({len(comuns)} conceitos)")
                    st.caption("Conceitos que permaneceram na sua estrutura, organizados por nível de abstração.")

                    if len(comuns) > 0:
                        # SEPARAÇÃO POR 6 NÍVEIS NATIVOS DO OPENALEX
                        levels_6 = {i: [] for i in range(6)}
                        indef = []
                        
                        for c in comuns:
                            if c in nodes_info:
                                try:
                                    lvl = int(float(nodes_info[c].get('level', -1)))
                                    if 0 <= lvl <= 5:
                                        levels_6[lvl].append(c)
                                    else:
                                        indef.append(c)
                                except:
                                    indef.append(c)
                            else:
                                indef.append(c)

                        # EXIBIÇÃO (MAPA OU LISTA)
                        tab_vis, tab_list = st.tabs(["🗺️ Mapa Hierárquico", "🔤 Lista Alfabética"])
                        
                        with tab_vis:
                            if len(indef) > len(comuns) * 0.8:
                                st.warning("⚠️ Dados históricos sem níveis hierárquicos suficientes.")
                                st.info("Use a aba 'Lista Alfabética' ao lado.")
                            else:
                                # Top N por nível (ordenados por score)
                                def top_by_score(lista, n=6):
                                    scored = [(c, nodes_info.get(c, {}).get('score', 0)) for c in lista]
                                    scored.sort(key=lambda x: x[1], reverse=True)
                                    return [c for c, _ in scored[:n]]
                                
                                def clean(s): return '"' + s.replace('"', "'").replace('\n', ' ') + '"'
                                
                                # Cores e labels para 6 níveis (gradiente azul)
                                cores = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb"]
                                font_cores = ["#1e3a5f", "#1e3a5f", "#1e3a5f", "#ffffff", "#ffffff", "#ffffff"]
                                labels = ["L0: Raiz", "L1: Área", "L2: Campo", "L3: Subcampo", "L4: Tópico", "L5: Específico"]
                                
                                graph_code = '''digraph {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, margin="0.15,0.08"];
    nodesep=0.3;
    ranksep=0.6;
    bgcolor="transparent";
'''
                                total_mostrado = 0
                                niveis_com_dados = []
                                
                                for lvl in range(6):
                                    top = top_by_score(levels_6[lvl])
                                    if top:
                                        niveis_com_dados.append(lvl)
                                        for c in top:
                                            label_node = f"{c}\\n({labels[lvl]})"
                                            graph_code += f'    {clean(c)} [fillcolor="{cores[lvl]}", fontcolor="{font_cores[lvl]}", label="{label_node}"];\n'
                                        graph_code += f'    {{ rank=same; {" ".join([clean(c) for c in top])} }}\n'
                                        total_mostrado += len(top)
                                
                                # Conexões entre níveis adjacentes que têm dados
                                for i in range(len(niveis_com_dados) - 1):
                                    lvl_atual = niveis_com_dados[i]
                                    lvl_prox = niveis_com_dados[i + 1]
                                    top_atual = top_by_score(levels_6[lvl_atual], 1)
                                    top_prox = top_by_score(levels_6[lvl_prox], 1)
                                    if top_atual and top_prox:
                                        graph_code += f'    {clean(top_atual[0])} -> {clean(top_prox[0])} [color="#94a3b8", style=dashed, arrowhead=none];\n'
                                
                                graph_code += "}"
                                
                                try:
                                    st.graphviz_chart(graph_code, width="stretch")
                                    st.caption(f"Exibindo top {total_mostrado} conceitos (de {len(comuns)}) por relevância. OpenAlex Level 0-5.")
                                except Exception as e:
                                    st.warning("⚠️ Não foi possível renderizar o mapa.")
                                    with st.expander("Erro técnico"):
                                        st.write(e)
                                    st.write(", ".join(sorted(comuns)[:30]) + "...")

                        with tab_list:
                            conceitos_ordenados = sorted(comuns)
                            if conceitos_ordenados:
                                num_colunas = 4
                                tamanho_fatia = -(-len(conceitos_ordenados) // num_colunas)
                                
                                cols = st.columns(num_colunas)
                                for i in range(num_colunas):
                                    with cols[i]:
                                        inicio = i * tamanho_fatia
                                        fim = inicio + tamanho_fatia
                                        sublista = conceitos_ordenados[inicio:fim]
                                        
                                        for conceito in sublista:
                                            st.markdown(f"<div style='margin-bottom:2px;'>• {conceito}</div>", unsafe_allow_html=True)
                            else:
                                st.write("A lista está vazia.")

                # ================== ANÁLISE PEDAGÓGICA ==================
                st.divider()
                st.markdown("### 🤖 O que o Delinéia diz sobre sua evolução?")
                
                if metrics['jaccard'] < 0.99:
                    # Mostrar resultado se já tiver
                    if 'ultima_analise_historico' in st.session_state:
                        st.markdown("### 📝 Parecer da Orientação Artificial")
                        st.info(st.session_state['ultima_analise_historico'], icon="🤖")
                        
                        # Botões de ação
                        col_pdf, col_novo = st.columns(2)
                        
                        with col_pdf:
                            try:
                                from pdf_generator import generate_comparison_pdf
                                
                                safe_df1 = st.session_state.get('df1_rico')
                                safe_df2 = st.session_state.get('df2_rico')
                                meta_antigo = getattr(safe_df1, 'attrs', {}).get('metadata', {}) if safe_df1 is not None else {}
                                meta_novo = getattr(safe_df2, 'attrs', {}).get('metadata', {}) if safe_df2 is not None else {}
                                
                                # Cache do PDF para evitar "Missing File"
                                if 'cache_pdf_historico' not in st.session_state:
                                    st.session_state.cache_pdf_historico = generate_comparison_pdf(
                                        form_data=st.session_state.get('form_data', {}),
                                        metrics=metrics,
                                        meta_antigo=meta_antigo,
                                        meta_novo=meta_novo,
                                        analise_ia=st.session_state['ultima_analise_historico'],
                                        nodes_info=nodes_info
                                    )
                                
                                nome_aluno_limpo = st.session_state.get('form_data', {}).get('nome', 'aluno').split()[0]
                                nome_arquivo = f"compara_grafos_{nome_aluno_limpo}.pdf"
                                
                                st.download_button(
                                    label="📥 Baixar Relatório PDF",
                                    data=st.session_state.cache_pdf_historico,
                                    file_name=nome_arquivo,
                                    mime="application/pdf",
                                    width="stretch",
                                    key="dl_historico_pdf"
                                )
                            except Exception as e:
                                st.warning(f"PDF indisponível: {e}")
                        
                        with col_novo:
                            if st.button("🔄 Novo Delineamento", key="btn_novo_delin", width="stretch", type="primary"):
                                # Limpa dados e volta ao formulário
                                st.session_state.resultado = None
                                st.session_state.form_data = {}
                                st.session_state.comparacao_ativa = False
                                if 'ultima_analise_historico' in st.session_state:
                                    del st.session_state['ultima_analise_historico']
                                st.rerun()

                    else:
                        # Mostrar botão para gerar
                        if st.button("✨ Gerar Análise Pedagógica da Mudança", type="primary", width="stretch", key="btn_analise_ia_tab3"):
                            # Preparação de variáveis básicas
                            nome_aluno = st.session_state.form_data.get('nome', 'Pesquisador').split()[0]
                            genero_aluno = st.session_state.form_data.get('genero', 'Neutro')
                            
                            with st.spinner(f"🧠 O Orientador Artificial está analisando a trajetória de {nome_aluno}..."):
                                try:
                                    # Garante que a instância do gerador existe
                                    if 'gemini_gen' not in st.session_state:
                                        from research_pipeline import GeminiQueryGenerator
                                        st.session_state.gemini_gen = GeminiQueryGenerator()
                                    
                                    # 1. EXTRAÇÃO DO CONTEXTO HISTÓRICO
                                    safe_df1 = st.session_state.get('df1_rico')
                                    safe_df2 = st.session_state.get('df2_rico')
                                    
                                    meta_antigo = getattr(safe_df1, 'attrs', {}).get('metadata', {}) if safe_df1 is not None else {}
                                    meta_novo = getattr(safe_df2, 'attrs', {}).get('metadata', {}) if safe_df2 is not None else {}

                                    # 2. CHAMADA DA NOVA FUNÇÃO CONTEXTUAL                                    
                                    analise = st.session_state.gemini_gen.generate_contextual_evolution_analysis(
                                        metrics=metrics,
                                        meta_antigo=meta_antigo,
                                        meta_novo=meta_novo,
                                        genero=genero_aluno
                                    )
                                    
                                    # 3. Salva e recarrega
                                    st.session_state['ultima_analise_historico'] = analise
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Erro na conexão com IA: {str(e)}")
                                    # Dica de debug: st.write(e)
                else:
                    st.info("Os dois delineamentos são idênticos.")

    rodape_institucional()

# ==================== ABA 4: PAINEL DE ANÁLISE ====================
with tab4:
    st.title("🔍 Painel de Exploração de Dados")
    st.caption("Análise profunda dos dados do OpenAlex")

    # Inicializar valores padrão no session_state
    if 'painel_limit' not in st.session_state:
        st.session_state.painel_limit = 500
    if 'painel_min_score' not in st.session_state:
        st.session_state.painel_min_score = 0.35
    if 'painel_min_level' not in st.session_state:
        st.session_state.painel_min_level = 0
    if 'painel_min_cooc' not in st.session_state:
        st.session_state.painel_min_cooc = 2

    with st.expander("🔍 Configurar Nova Busca", expanded=False):
        # Sincroniza dashboard_query com o campo de texto
        if 'dashboard_query' in st.session_state and st.session_state.dashboard_query:
            # Se dashboard_query foi atualizado, propaga para o campo de texto
            if st.session_state.get('txt_query_painel') != st.session_state.dashboard_query:
                st.session_state.txt_query_painel = st.session_state.dashboard_query
        
        # Inicializa o valor se não existir
        if 'txt_query_painel' not in st.session_state:
            st.session_state.txt_query_painel = st.session_state.get('dashboard_query', "games AND education")
        
        # Campo de busca
        query = st.text_input(
            "Chave de Busca:",
            help="Use operadores: AND, OR, NOT",
            key="txt_query_painel"
        )

        if 'dashboard_query' in st.session_state and st.session_state.dashboard_query:
            source = st.session_state.get('dashboard_query_source', 'delineascópio')
            if source == "construtor":
                st.info("📋 Chave de busca copiada do construtor de chaves")
            else:
                st.info("📋 Chave de busca copiada do Delineascópio")

        st.subheader("🔧 Filtros")

        with st.expander("⚙️ Configurações Avançadas", expanded=True):
            sync_config = st.checkbox("Usar configuração padrão", value=True, key="sync_config_painel")

            if sync_config:
                st.info("**Configuração Padrão:**\n- Limite: 500 artigos\n- Score mínimo: 0.35\n- Level mínimo: 0")
                st.session_state.painel_limit = 500
                st.session_state.painel_min_score = 0.35
                st.session_state.painel_min_level = 0
                st.session_state.painel_min_cooc = 1
            else:
                st.session_state.painel_limit = st.slider(
                    "Limite de artigos:", 10, 500, st.session_state.painel_limit, 10,
                    help="Número máximo de artigos a buscar na API OpenAlex",
                    key="slider_limit_painel"
                )
                st.session_state.painel_min_score = st.slider(
                    "Score mínimo:", 0.0, 1.0, st.session_state.painel_min_score, 0.05,
                    help="Relevância mínima do conceito (0-1). Valores maiores = conceitos mais relevantes",
                    key="slider_score_painel"
                )
                st.session_state.painel_min_level = st.slider(
                    "Level mínimo:", 0, 5, st.session_state.painel_min_level, 1,
                    help="Nível hierárquico do conceito (0-5). 0 = geral, 5 = muito específico",
                    key="slider_level_painel"
                )
                st.session_state.painel_min_cooc = st.slider (
                    "Coocorrência mínima:", 1, 10, st.session_state.painel_min_cooc, 2,
                    help="Frequência mínima de coocorrência para formar aresta no grafo",
                    key="slider_cooc_painel"
                )

        # Variáveis locais para uso posterior
        limit = st.session_state.painel_limit
        min_score = st.session_state.painel_min_score
        min_level = st.session_state.painel_min_level
        min_cooc = st.session_state.painel_min_cooc

        st.divider()

        # Botão de buscar
        if st.button("🔍 Buscar", type="primary", width="stretch", key="btn_buscar_painel"):
            limpar_memoria()
            with st.spinner("🔄 Em processamento... confira no Painel"):
                try:
                    # Usar valores do session_state
                    limit = st.session_state.painel_limit
                    min_score = st.session_state.painel_min_score
                    min_level = st.session_state.painel_min_level

                    # 1. BUSCA (CACHEADA)
                    raw_articles = search_openalex_cached(query, limit, 0, 0)
                    
                    # 2. FILTRAGEM LOCAL (PYTHON)
                    filtered_concepts_lists = []
                    
                    for article in raw_articles:
                        concepts = [
                            c.get('display_name', c.get('name')) 
                            for c in article.get('concepts', [])
                            if c.get('score', 0) >= min_score and c.get('level', 0) >= min_level
                        ]
                        if concepts:
                            filtered_concepts_lists.append(concepts)

                    # 3. CONSTRUÇÃO DO GRAFO
                    analyzer = CooccurrenceAnalyzer()
                    G = analyzer.build_graph(filtered_concepts_lists, min_cooc)

                    # 4. DATAFRAME LIMPO
                    df_display = process_openalex_dataframe(raw_articles)

                    # Salvar no Session State
                    st.session_state.dashboard_data = {
                        'articles': raw_articles,
                        'df_display': df_display,
                        'concepts_lists': filtered_concepts_lists,
                        'graph': G
                    }

                    with st.expander("📋 Detalhes da Busca", expanded=True):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.write(f"**Chave:** `{query}`")
                            st.write(f"**Artigos:** {len(raw_articles)}")
                        with col_d2:
                            st.write(f"**Filtros:** Score ≥ {min_score} | Level ≥ {min_level}")
                            st.write(f"**Nós no grafo:** {len(G.nodes())}")

                    st.success(f"✅ Análise concluída: {len(raw_articles)} artigos | {len(G.nodes())} nós no grafo")

                except Exception as e:
                    st.error(f"❌ Erro na busca: {str(e)}")
   
    # Área principal do painel
    # Verifica se TEM dados antes de tentar ler
    if st.session_state.dashboard_data is not None:
        # Recuperar dados
        data = st.session_state.dashboard_data
        articles = data['articles']
        concepts_lists = data['concepts_lists']
        G = data['graph']

        # Criar sub-abas para análises (Adicionei "📜 Histórico")
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📚 Artigos", "🧩 Conceitos", "🔗 Coocorrências", 
            "🕸️ Grafo", "🗺️ Mapa Temático", "📊 Estatísticas", 
            "💾 Exportação"
        ])

        # ========== SUB-ABA 1: ARTIGOS ==========
        with t1:
            st.header("📚 Artigos")
            
            # Recupera o DataFrame limpo que geramos no botão Buscar
            # Isso evita ter que recriar o DataFrame a cada clique na aba
            df_display = st.session_state.dashboard_data.get('df_display')
            
            if df_display is not None and not df_display.empty:
                st.metric("Total de Artigos", len(df_display))
                
                # Exibe Tabela Interativa (índice começando em 1)
                df_display_indexed = df_display.copy()
                df_display_indexed.index = range(1, len(df_display_indexed) + 1)
                
                st.dataframe(
                    df_display_indexed,
                    width="stretch",
                    height=400,
                    column_config={
                        "DOI/URL": st.column_config.LinkColumn(
                            "🔗 Link",
                            help="Acesse o artigo original",
                            display_text="Abrir Artigo"
                        ),
                        "Citações": st.column_config.NumberColumn(
                            "Citações",
                            format="%d ⭐"
                        )
                    }
                )
            else:
                st.warning("Nenhum dado para exibir ou busca ainda não realizada.")

            # Detalhes do Artigo
    
            if len(articles) > 0:
                st.divider()
                st.subheader("🔍 Detalhes do Artigo")
    
                # Seletor de artigo - CORRIGIDO
                idx = st.selectbox(
                    "Selecione um artigo:",
                    range(len(articles)),
                    format_func=lambda i: f"{i+1}. {(articles[i].get('title') or 'Sem título')[:60]}..."
                )
    
                selected = articles[idx]
    
                col1, col2 = st.columns([2, 1])
    
                with col1:
                    st.write(f"**Título:** {selected.get('title') or 'Sem título'}")
                    st.write(f"**Ano:** {selected.get('year', 'N/A')}")
    
                    # ✨ EXIBIR LINK CLICÁVEL ✨
                    link = selected.get('doi', selected.get('url', ''))
                    if link:
                        st.markdown(f"**🔗 Link:** [{link}]({link})")
                    else:
                        st.write("**🔗 Link:** N/A")
    
                with col2:
                    st.metric("Conceitos", len(selected.get('concepts', [])))
    
                st.subheader("📋 Conceitos do Artigo")
    
                concepts_df = pd.DataFrame([
                    {
                        # Usa display_name ou name, igual fizemos no pipeline
                        'Conceito': c.get('display_name', c.get('name', 'Sem nome')), 
                        'Score': f"{c.get('score', 0):.3f}",
                        'Level': c.get('level', '?')
                    }
                    for c in selected.get('concepts', [])
                ])
    
                if not concepts_df.empty:
                    st.dataframe(concepts_df, width="stretch")
                else:
                    st.info("Nenhum conceito encontrado")
    
                with st.expander("🔍 Ver JSON completo"):
                    st.json(selected)

            # ========== ANÁLISE DE PRODUÇÃO ANUAL ==========
            if df_display is not None and not df_display.empty and 'Ano' in df_display.columns:
                st.divider()
                st.subheader("📈 Distribuição Anual da Produção")
                
                # Extrair anos e contar frequências
                anos_validos = df_display['Ano'].dropna()
                anos_validos = anos_validos[anos_validos != ''].astype(int)
                
                if len(anos_validos) > 0:
                    # Criar DataFrame de frequências por ano
                    freq_anos = anos_validos.value_counts().sort_index()
                    
                    # Criar DataFrame completo com todos os anos no intervalo
                    ano_min, ano_max = freq_anos.index.min(), freq_anos.index.max()
                    todos_anos = range(ano_min, ano_max + 1)
                    
                    df_anos = pd.DataFrame({
                        'Ano': list(todos_anos),
                        'Frequência Absoluta': [freq_anos.get(ano, 0) for ano in todos_anos]
                    })
                    
                    # Calcular métricas
                    total = df_anos['Frequência Absoluta'].sum()
                    df_anos['Frequência Relativa (%)'] = (df_anos['Frequência Absoluta'] / total * 100).round(2)
                    df_anos['Freq. Acumulada Absoluta'] = df_anos['Frequência Absoluta'].cumsum()
                    df_anos['Freq. Acumulada Relativa (%)'] = (df_anos['Freq. Acumulada Absoluta'] / total * 100).round(2)
                    
                    # Taxa de crescimento (% em relação ao ano anterior)
                    df_anos['Taxa de Crescimento (%)'] = df_anos['Frequência Absoluta'].pct_change().fillna(0) * 100
                    # Substituir infinitos por 0 (quando ano anterior tinha 0 publicações)
                    df_anos['Taxa de Crescimento (%)'] = df_anos['Taxa de Crescimento (%)'].replace([np.inf, -np.inf], 0).round(2)
                    
                    # Gráfico de linha
                    fig_anos = go.Figure()
                    
                    # Linha azul - Frequência absoluta
                    fig_anos.add_trace(go.Scatter(
                        x=df_anos['Ano'],
                        y=df_anos['Frequência Absoluta'],
                        mode='lines+markers',
                        name='Nº de Artigos',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Linha de tendência (média móvel 3 anos)
                    if len(df_anos) >= 3:
                        df_anos['Tendência'] = df_anos['Frequência Absoluta'].rolling(window=3, min_periods=1).mean()
                        fig_anos.add_trace(go.Scatter(
                            x=df_anos['Ano'],
                            y=df_anos['Tendência'],
                            mode='lines',
                            name='Tendência (média móvel)',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                    
                    fig_anos.update_layout(
                        title='Distribuição Anual de Publicações',
                        xaxis_title='Ano de Publicação',
                        yaxis_title='Número de Artigos',
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )
                    
                    st.plotly_chart(fig_anos, width='stretch')
                    
                    # Tabela de frequências
                    st.markdown("**📋 Tabela de Frequências por Ano**")
                    
                    # Preparar tabela para exibição
                    df_tabela = df_anos[['Ano', 'Frequência Absoluta', 'Frequência Relativa (%)', 
                                         'Freq. Acumulada Absoluta', 'Freq. Acumulada Relativa (%)',
                                         'Taxa de Crescimento (%)']].copy()
                    df_tabela = df_tabela.rename(columns={'Ano': 'Ano de Publicação'})
                    df_tabela = df_tabela.set_index('Ano de Publicação')
                    
                    st.dataframe(df_tabela, width='stretch')
                    
                    # Métricas resumo
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Período", f"{ano_min} - {ano_max}")
                    col_m2.metric("Ano com mais publicações", f"{df_anos.loc[df_anos['Frequência Absoluta'].idxmax(), 'Ano']}")
                    
                    # Taxa média de crescimento (excluindo primeiro ano e infinitos)
                    taxas_validas = df_anos['Taxa de Crescimento (%)'].iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()
                    taxa_media = taxas_validas.mean() if len(taxas_validas) > 0 else 0
                    col_m3.metric("Taxa média de crescimento", f"{taxa_media:.1f}%")
                else:
                    st.info("Dados de ano não disponíveis para análise temporal.")

            # ========== ANÁLISE DE DISTRIBUIÇÃO POR PAÍSES ==========
            if len(articles) > 0:
                st.divider()
                st.subheader("🌍 Distribuição Geográfica")
                
                # Extrair países das instituições dos autores
                from collections import Counter
                
                paises_counter = Counter()
                
                for article in articles:
                    authorships = article.get('authorships', [])
                    paises_artigo = set()  # Evita contar o mesmo país várias vezes por artigo
                    
                    for authorship in authorships:
                        institutions = authorship.get('institutions', [])
                        for inst in institutions:
                            country_code = inst.get('country_code')
                            if country_code:
                                paises_artigo.add(country_code)
                    
                    for pais in paises_artigo:
                        paises_counter[pais] += 1
                
                if paises_counter:
                    # Criar DataFrame
                    df_paises = pd.DataFrame(
                        paises_counter.most_common(),
                        columns=['Código', 'Artigos']
                    )
                    
                    # Dicionário de nomes de países (ISO Alpha-2 para nome)
                    nomes_paises = {
                        # África
                        'DZ': 'Argélia', 'AO': 'Angola', 'BJ': 'Benin', 'BW': 'Botsuana',
                        'BF': 'Burkina Faso', 'BI': 'Burundi', 'CV': 'Cabo Verde', 'CM': 'Camarões',
                        'CF': 'República Centro-Africana', 'TD': 'Chade', 'KM': 'Comores',
                        'CG': 'Congo', 'CD': 'República Democrática do Congo', 'CI': 'Costa do Marfim',
                        'DJ': 'Djibuti', 'EG': 'Egito', 'GQ': 'Guiné Equatorial', 'ER': 'Eritreia',
                        'SZ': 'Essuatíni', 'ET': 'Etiópia', 'GA': 'Gabão', 'GM': 'Gâmbia',
                        'GH': 'Gana', 'GN': 'Guiné', 'GW': 'Guiné-Bissau', 'KE': 'Quênia',
                        'LS': 'Lesoto', 'LR': 'Libéria', 'LY': 'Líbia', 'MG': 'Madagascar',
                        'MW': 'Maláui', 'ML': 'Mali', 'MR': 'Mauritânia', 'MU': 'Maurício',
                        'MA': 'Marrocos', 'MZ': 'Moçambique', 'NA': 'Namíbia', 'NE': 'Níger',
                        'NG': 'Nigéria', 'RW': 'Ruanda', 'ST': 'São Tomé e Príncipe', 'SN': 'Senegal',
                        'SC': 'Seicheles', 'SL': 'Serra Leoa', 'SO': 'Somália', 'ZA': 'África do Sul',
                        'SS': 'Sudão do Sul', 'SD': 'Sudão', 'TZ': 'Tanzânia', 'TG': 'Togo',
                        'TN': 'Tunísia', 'UG': 'Uganda', 'ZM': 'Zâmbia', 'ZW': 'Zimbábue',
                        
                        # Américas
                        'AG': 'Antígua e Barbuda', 'AR': 'Argentina', 'BS': 'Bahamas', 'BB': 'Barbados',
                        'BZ': 'Belize', 'BO': 'Bolívia', 'BR': 'Brasil', 'CA': 'Canadá',
                        'CL': 'Chile', 'CO': 'Colômbia', 'CR': 'Costa Rica', 'CU': 'Cuba',
                        'DM': 'Dominica', 'DO': 'República Dominicana', 'EC': 'Equador',
                        'SV': 'El Salvador', 'GD': 'Granada', 'GT': 'Guatemala', 'GY': 'Guiana',
                        'HT': 'Haiti', 'HN': 'Honduras', 'JM': 'Jamaica', 'MX': 'México',
                        'NI': 'Nicarágua', 'PA': 'Panamá', 'PY': 'Paraguai', 'PE': 'Peru',
                        'KN': 'São Cristóvão e Névis', 'LC': 'Santa Lúcia', 'VC': 'São Vicente e Granadinas',
                        'SR': 'Suriname', 'TT': 'Trinidad e Tobago', 'US': 'Estados Unidos',
                        'UY': 'Uruguai', 'VE': 'Venezuela', 'PR': 'Porto Rico',
                        
                        # Ásia
                        'AF': 'Afeganistão', 'AM': 'Armênia', 'AZ': 'Azerbaijão', 'BH': 'Bahrein',
                        'BD': 'Bangladesh', 'BT': 'Butão', 'BN': 'Brunei', 'KH': 'Camboja',
                        'CN': 'China', 'CY': 'Chipre', 'GE': 'Geórgia', 'IN': 'Índia',
                        'ID': 'Indonésia', 'IR': 'Irã', 'IQ': 'Iraque', 'IL': 'Israel',
                        'JP': 'Japão', 'JO': 'Jordânia', 'KZ': 'Cazaquistão', 'KW': 'Kuwait',
                        'KG': 'Quirguistão', 'LA': 'Laos', 'LB': 'Líbano', 'MY': 'Malásia',
                        'MV': 'Maldivas', 'MN': 'Mongólia', 'MM': 'Mianmar', 'NP': 'Nepal',
                        'KP': 'Coreia do Norte', 'OM': 'Omã', 'PK': 'Paquistão', 'PS': 'Palestina',
                        'PH': 'Filipinas', 'QA': 'Catar', 'SA': 'Arábia Saudita', 'SG': 'Singapura',
                        'KR': 'Coreia do Sul', 'LK': 'Sri Lanka', 'SY': 'Síria', 'TW': 'Taiwan',
                        'TJ': 'Tajiquistão', 'TH': 'Tailândia', 'TL': 'Timor-Leste', 'TR': 'Turquia',
                        'TM': 'Turcomenistão', 'AE': 'Emirados Árabes Unidos', 'UZ': 'Uzbequistão',
                        'VN': 'Vietnã', 'YE': 'Iêmen', 'HK': 'Hong Kong', 'MO': 'Macau',
                        
                        # Europa
                        'AL': 'Albânia', 'AD': 'Andorra', 'AT': 'Áustria', 'BY': 'Bielorrússia',
                        'BE': 'Bélgica', 'BA': 'Bósnia e Herzegovina', 'BG': 'Bulgária', 'HR': 'Croácia',
                        'CZ': 'República Tcheca', 'DK': 'Dinamarca', 'EE': 'Estônia', 'FI': 'Finlândia',
                        'FR': 'França', 'DE': 'Alemanha', 'GR': 'Grécia', 'HU': 'Hungria',
                        'IS': 'Islândia', 'IE': 'Irlanda', 'IT': 'Itália', 'XK': 'Kosovo',
                        'LV': 'Letônia', 'LI': 'Liechtenstein', 'LT': 'Lituânia', 'LU': 'Luxemburgo',
                        'MT': 'Malta', 'MD': 'Moldávia', 'MC': 'Mônaco', 'ME': 'Montenegro',
                        'NL': 'Países Baixos', 'MK': 'Macedônia do Norte', 'NO': 'Noruega', 'PL': 'Polônia',
                        'PT': 'Portugal', 'RO': 'Romênia', 'RU': 'Rússia', 'SM': 'San Marino',
                        'RS': 'Sérvia', 'SK': 'Eslováquia', 'SI': 'Eslovênia', 'ES': 'Espanha',
                        'SE': 'Suécia', 'CH': 'Suíça', 'UA': 'Ucrânia', 'GB': 'Reino Unido',
                        'VA': 'Vaticano',
                        
                        # Oceania
                        'AU': 'Austrália', 'FJ': 'Fiji', 'KI': 'Kiribati', 'MH': 'Ilhas Marshall',
                        'FM': 'Micronésia', 'NR': 'Nauru', 'NZ': 'Nova Zelândia', 'PW': 'Palau',
                        'PG': 'Papua-Nova Guiné', 'WS': 'Samoa', 'SB': 'Ilhas Salomão', 'TO': 'Tonga',
                        'TV': 'Tuvalu', 'VU': 'Vanuatu', 'NC': 'Nova Caledônia', 'PF': 'Polinésia Francesa',
                        'GU': 'Guam'
                    }
                    
                    df_paises['País'] = df_paises['Código'].map(
                        lambda x: nomes_paises.get(x, x)
                    )
                    
                    # Converter códigos ISO Alpha-2 para Alpha-3
                    iso2_to_iso3 = {
                        'AF': 'AFG', 'AL': 'ALB', 'DZ': 'DZA', 'AD': 'AND', 'AO': 'AGO',
                        'AG': 'ATG', 'AR': 'ARG', 'AM': 'ARM', 'AU': 'AUS', 'AT': 'AUT',
                        'AZ': 'AZE', 'BS': 'BHS', 'BH': 'BHR', 'BD': 'BGD', 'BB': 'BRB',
                        'BY': 'BLR', 'BE': 'BEL', 'BZ': 'BLZ', 'BJ': 'BEN', 'BT': 'BTN',
                        'BO': 'BOL', 'BA': 'BIH', 'BW': 'BWA', 'BR': 'BRA', 'BN': 'BRN',
                        'BG': 'BGR', 'BF': 'BFA', 'BI': 'BDI', 'CV': 'CPV', 'KH': 'KHM',
                        'CM': 'CMR', 'CA': 'CAN', 'CF': 'CAF', 'TD': 'TCD', 'CL': 'CHL',
                        'CN': 'CHN', 'CO': 'COL', 'KM': 'COM', 'CG': 'COG', 'CD': 'COD',
                        'CR': 'CRI', 'CI': 'CIV', 'HR': 'HRV', 'CU': 'CUB', 'CY': 'CYP',
                        'CZ': 'CZE', 'DK': 'DNK', 'DJ': 'DJI', 'DM': 'DMA', 'DO': 'DOM',
                        'EC': 'ECU', 'EG': 'EGY', 'SV': 'SLV', 'GQ': 'GNQ', 'ER': 'ERI',
                        'EE': 'EST', 'SZ': 'SWZ', 'ET': 'ETH', 'FJ': 'FJI', 'FI': 'FIN',
                        'FR': 'FRA', 'GA': 'GAB', 'GM': 'GMB', 'GE': 'GEO', 'DE': 'DEU',
                        'GH': 'GHA', 'GR': 'GRC', 'GD': 'GRD', 'GT': 'GTM', 'GN': 'GIN',
                        'GW': 'GNB', 'GY': 'GUY', 'HT': 'HTI', 'HN': 'HND', 'HU': 'HUN',
                        'IS': 'ISL', 'IN': 'IND', 'ID': 'IDN', 'IR': 'IRN', 'IQ': 'IRQ',
                        'IE': 'IRL', 'IL': 'ISR', 'IT': 'ITA', 'JM': 'JAM', 'JP': 'JPN',
                        'JO': 'JOR', 'KZ': 'KAZ', 'KE': 'KEN', 'KI': 'KIR', 'KP': 'PRK',
                        'KR': 'KOR', 'KW': 'KWT', 'KG': 'KGZ', 'LA': 'LAO', 'LV': 'LVA',
                        'LB': 'LBN', 'LS': 'LSO', 'LR': 'LBR', 'LY': 'LBY', 'LI': 'LIE',
                        'LT': 'LTU', 'LU': 'LUX', 'MG': 'MDG', 'MW': 'MWI', 'MY': 'MYS',
                        'MV': 'MDV', 'ML': 'MLI', 'MT': 'MLT', 'MH': 'MHL', 'MR': 'MRT',
                        'MU': 'MUS', 'MX': 'MEX', 'FM': 'FSM', 'MD': 'MDA', 'MC': 'MCO',
                        'MN': 'MNG', 'ME': 'MNE', 'MA': 'MAR', 'MZ': 'MOZ', 'MM': 'MMR',
                        'NA': 'NAM', 'NR': 'NRU', 'NP': 'NPL', 'NL': 'NLD', 'NZ': 'NZL',
                        'NI': 'NIC', 'NE': 'NER', 'NG': 'NGA', 'MK': 'MKD', 'NO': 'NOR',
                        'OM': 'OMN', 'PK': 'PAK', 'PW': 'PLW', 'PS': 'PSE', 'PA': 'PAN',
                        'PG': 'PNG', 'PY': 'PRY', 'PE': 'PER', 'PH': 'PHL', 'PL': 'POL',
                        'PT': 'PRT', 'QA': 'QAT', 'RO': 'ROU', 'RU': 'RUS', 'RW': 'RWA',
                        'KN': 'KNA', 'LC': 'LCA', 'VC': 'VCT', 'WS': 'WSM', 'SM': 'SMR',
                        'ST': 'STP', 'SA': 'SAU', 'SN': 'SEN', 'RS': 'SRB', 'SC': 'SYC',
                        'SL': 'SLE', 'SG': 'SGP', 'SK': 'SVK', 'SI': 'SVN', 'SB': 'SLB',
                        'SO': 'SOM', 'ZA': 'ZAF', 'SS': 'SSD', 'ES': 'ESP', 'LK': 'LKA',
                        'SD': 'SDN', 'SR': 'SUR', 'SE': 'SWE', 'CH': 'CHE', 'SY': 'SYR',
                        'TW': 'TWN', 'TJ': 'TJK', 'TZ': 'TZA', 'TH': 'THA', 'TL': 'TLS',
                        'TG': 'TGO', 'TO': 'TON', 'TT': 'TTO', 'TN': 'TUN', 'TR': 'TUR',
                        'TM': 'TKM', 'TV': 'TUV', 'UG': 'UGA', 'UA': 'UKR', 'AE': 'ARE',
                        'GB': 'GBR', 'US': 'USA', 'UY': 'URY', 'UZ': 'UZB', 'VU': 'VUT',
                        'VA': 'VAT', 'VE': 'VEN', 'VN': 'VNM', 'YE': 'YEM', 'ZM': 'ZMB',
                        'ZW': 'ZWE', 'HK': 'HKG', 'MO': 'MAC', 'PR': 'PRI', 'XK': 'XKX'
                    }
                    
                    df_paises['ISO3'] = df_paises['Código'].map(iso2_to_iso3)
                    
                    # Mapa geográfico (Choropleth)
                    fig_mapa = px.choropleth(
                        df_paises,
                        locations='ISO3',
                        locationmode='ISO-3',
                        color='Artigos',
                        hover_name='País',
                        hover_data={'Código': False, 'Artigos': True},
                        color_continuous_scale='Blues',
                        title='Distribuição de Artigos por País'
                    )
                    
                    fig_mapa.update_layout(
                        height=500,
                        geo=dict(
                            showframe=False,
                            showcoastlines=True,
                            projection_type='natural earth'
                        ),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                    
                    st.plotly_chart(fig_mapa, width='stretch')
                    
                    # Métricas
                    col_g1, col_g2, col_g3 = st.columns(3)
                    col_g1.metric("Países representados", len(df_paises))
                    col_g2.metric("País líder", df_paises.iloc[0]['País'])
                    col_g3.metric("Artigos do líder", df_paises.iloc[0]['Artigos'])
                    
                    # Tabela com ranking
                    st.markdown("**📋 Ranking de Países**")
                    
                    df_tabela_paises = df_paises[['País', 'Código', 'Artigos']].copy()
                    df_tabela_paises['% do Total'] = (df_tabela_paises['Artigos'] / df_tabela_paises['Artigos'].sum() * 100).round(2)
                    df_tabela_paises.index = range(1, len(df_tabela_paises) + 1)
                    df_tabela_paises.index.name = 'Rank'
                    
                    st.dataframe(df_tabela_paises, width='stretch')
                else:
                    st.info("Dados de país não disponíveis para os artigos recuperados.")

        # ========== SUB-ABA 2: CONCEITOS ==========
        with t2:
            st.header("🧩 Conceitos")

            # Estatísticas gerais
            all_concepts = [c for cl in concepts_lists for c in cl]
            freq = Counter(all_concepts)

            col1, col2, col3 = st.columns(3)
            col1.metric("Artigos com Conceitos", len(concepts_lists))
            col2.metric("Conceitos Únicos", len(freq))
            col3.metric("Total de Ocorrências", len(all_concepts))

            st.divider()

            # ===== NUVEM DE PALAVRAS (com Plotly) =====
            st.subheader("☁️ Nuvem de Conceitos")
            
            # Criar dicionário de frequências
            freq_dict = dict(freq.most_common(50))
            
            if freq_dict:
                import random
                random.seed(42)
                
                # Preparar dados
                words = list(freq_dict.keys())
                frequencies = list(freq_dict.values())
                max_freq = max(frequencies)
                min_freq = min(frequencies)
                
                # Normalizar tamanhos (entre 12 e 80)
                sizes = [12 + (f - min_freq) / (max_freq - min_freq) * 68 if max_freq > min_freq else 40 for f in frequencies]
                
                # Posições em espiral/orgânica
                n = len(words)
                x_positions = []
                y_positions = []
                for i in range(n):
                    angle = i * 2.4  # Ângulo áureo
                    radius = 10 + i * 1.5
                    x_positions.append(50 + radius * np.cos(angle) * 0.8)
                    y_positions.append(50 + radius * np.sin(angle) * 0.5)
                
                # Paleta de cores mais harmônica
                color_palette = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#e9c46a', 
                                '#023e8a', '#0077b6', '#8338ec', '#ff006e', '#06d6a0']
                colors = [color_palette[i % len(color_palette)] for i in range(n)]
                
                # Criar figura
                fig_cloud = go.Figure()
                
                for i, word in enumerate(words):
                    fig_cloud.add_trace(go.Scatter(
                        x=[x_positions[i]],
                        y=[y_positions[i]],
                        mode='text',
                        text=[word],
                        textfont=dict(size=sizes[i], color=colors[i], family='Arial Black'),
                        hoverinfo='text',
                        hovertext=f'{word}: {frequencies[i]} ocorrências',
                        showlegend=False
                    ))
                
                fig_cloud.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
                    height=450,
                    margin=dict(l=0, r=0, t=10, b=10),
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig_cloud, width="stretch")
            else:
                st.info("Sem dados suficientes para gerar nuvem de palavras")
            
            st.divider()

            # Top conceitos
            st.subheader("🏆 Conceitos Mais Frequentes")

            top_n = st.slider("Número de conceitos:", 10, 50, 20, 5, key="top_concepts")

            df_freq = pd.DataFrame(
                freq.most_common(top_n),
                columns=['Conceito', 'Frequência']
            )

            # Tabela primeiro
            st.markdown("**📋 Tabela de Frequências**")
            st.dataframe(df_freq, width="stretch")
            
            st.divider()

            # Gráfico de barras depois
            fig = px.bar(
                df_freq,
                x='Frequência',
                y='Conceito',
                orientation='h',
                title=f"Top {top_n} Conceitos Mais Frequentes",
                color='Frequência',
                color_continuous_scale='blues'
            )
            fig.update_layout(
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, width="stretch")

            # Análise de Zipf
            @st.cache_data
            def cached_zipf_analysis(frequency_data):
                """Wrapper para cachear a análise de Zipf."""
                return analyze_zipf(frequency_data)
                """
                Analisa a distribuição de frequências segundo a Lei de Zipf

                Args:
                    frequency_data: Lista de tuplas (palavra, frequência) ordenada por frequência

                Returns:
                    dict com métricas e dados para plotagem
                """
                # Extrair frequências
                frequencies = [freq for _, freq in frequency_data]

                # Criar ranks (1, 2, 3, ...)
                ranks = np.arange(1, len(frequencies) + 1)

                # Converter para arrays numpy
                ranks_array = np.array(ranks)
                freq_array = np.array(frequencies)

                # Aplicar log para análise linear
                log_ranks = np.log10(ranks_array)
                log_freqs = np.log10(freq_array)

                # Regressão linear no espaço log-log
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)

                # Calcular R²
                r_squared = r_value ** 2

                # Gerar linha de tendência
                trend_line = 10 ** (slope * log_ranks + intercept)

                # Interpretação
                if r_squared > 0.90:
                    interpretation = "✅ Forte aderência à Lei de Zipf"
                    quality = "excelente"
                elif r_squared > 0.75:
                    interpretation = "⚠️ Aderência moderada à Lei de Zipf"
                    quality = "boa"
                else:
                    interpretation = "❌ Fraca aderência à Lei de Zipf"
                    quality = "baixa"

                # Análise da inclinação
                if -1.2 < slope < -0.8:
                    slope_interpretation = "próximo ao ideal (-1.0)"
                elif slope < -1.2:
                    slope_interpretation = "vocabulário mais concentrado que o esperado"
                else:
                    slope_interpretation = "vocabulário mais disperso que o esperado"

                return {
                    'ranks': ranks_array,
                    'frequencies': freq_array,
                    'log_ranks': log_ranks,
                    'log_freqs': log_freqs,
                    'trend_line': trend_line,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'interpretation': interpretation,
                    'quality': quality,
                    'slope_interpretation': slope_interpretation
                }

            # Executar análise de Zipf
            if len(freq) > 0:
                st.divider()
                st.subheader("📈 Análise da Lei de Zipf")

                st.markdown("""
                A **Lei de George Kingsley Zipf** prediz que a frequência de uma palavra é inversamente proporcional
                ao seu ranking. Em um gráfico log-log, isso aparece como uma linha reta com inclinação
                próxima a -1.0.
                """)

                # Preparar dados (tuplas de palavra, frequência)
                frequency_data = freq.most_common()

                # Chamar a função de análise
                zipf_results = analyze_zipf(frequency_data)

                # Exibir métricas
                col1, col2, col3 = st.columns(3)
                col1.metric("R² (Aderência)", f"{zipf_results['r_squared']:.3f}")
                col2.metric("Inclinação", f"{zipf_results['slope']:.3f}")
                col3.metric("Qualidade", zipf_results['quality'].upper())

                # Interpretações
                st.info(f"**{zipf_results['interpretation']}** - Inclinação {zipf_results['slope_interpretation']}")

                # Gráfico log-log
                fig_zipf = go.Figure()

                # Dados reais
                fig_zipf.add_trace(go.Scatter(
                    x=zipf_results['ranks'],
                    y=zipf_results['frequencies'],
                    mode='markers',
                    name='Dados Observados',
                    marker=dict(size=8, color='blue'),
                    text=[word for word, _ in frequency_data],
                    hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Frequência: %{y}<extra></extra>'
                ))

                # Linha de tendência (Lei de Zipf)
                fig_zipf.add_trace(go.Scatter(
                    x=zipf_results['ranks'],
                    y=zipf_results['trend_line'],
                    mode='lines',
                    name='Lei de Zipf (teórico)',
                    line=dict(color='red', dash='dash', width=2)
                ))

                fig_zipf.update_layout(
                    title='Distribuição de Zipf (Escala Log-Log)',
                    xaxis_title='Ranking (log)',
                    yaxis_title='Frequência (log)',
                    xaxis_type='log',
                    yaxis_type='log',
                    height=500,
                    hovermode='closest'
                )

                st.plotly_chart(fig_zipf, width="stretch")

                # Explicação adicional
                with st.expander("ℹ️ Como interpretar"):
                    st.markdown(f"""
                    **R² = {zipf_results['r_squared']:.3f}**
                    - R² > 0.90: Excelente aderência à Lei de Zipf
                    - 0.75 < R² < 0.90: Boa aderência
                    - R² < 0.75: Baixa aderência

                    **Inclinação = {zipf_results['slope']:.3f}**
                    - Ideal: próximo a -1.0
                    - Mais negativo: vocabulário concentrado em poucas palavras
                    - Menos negativo: vocabulário mais distribuído

                    **Significância estatística**: p-value = {zipf_results['p_value']:.6f}

                    **Referência**  
                    - ZIPF, G.K. Human behavior and the principle of least effort: an introduction to human ecology. Cambridge: Addison-Wesley Press, 1949. Disponível em: https://archive.org/details/in.ernet.dli.2015.90211.
                    """)

            # Distribuição
            st.subheader("📊 Distribuição de Conceitos por Artigo")

            concepts_per_article = [len(c) for c in concepts_lists]

            fig2 = px.histogram(
                x=concepts_per_article,
                nbins=20,
                labels={'x': 'Número de conceitos', 'y': 'Frequência'},
                title="Distribuição de Conceitos por Artigo"
            )

            st.plotly_chart(fig2, width="stretch")

            if len(concepts_per_article) > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("Média", f"{sum(concepts_per_article)/len(concepts_per_article):.1f}")
                col2.metric("Mínimo", min(concepts_per_article))
                col3.metric("Máximo", max(concepts_per_article))

            # ========== EVOLUÇÃO TEMPORAL DOS CONCEITOS ==========
            st.divider()
            st.subheader("📈 Evolução dos Conceitos ao Longo do Tempo")
            
            # Extrair conceitos por ano
            conceito_ano = {}  # {conceito: {ano: frequência}}
            
            for article in articles:
                ano = article.get('year')
                if not ano:
                    continue
                    
                concepts = article.get('concepts', [])
                for c in concepts:
                    nome = c.get('display_name', c.get('name', ''))
                    score = c.get('score', 0)
                    
                    # Filtrar por score mínimo
                    if nome and score >= 0.35:
                        if nome not in conceito_ano:
                            conceito_ano[nome] = {}
                        if ano not in conceito_ano[nome]:
                            conceito_ano[nome][ano] = 0
                        conceito_ano[nome][ano] += 1
            
            if conceito_ano:
                # Calcular total por conceito e selecionar top N
                top_n_temporal = st.slider(
                    "Número de conceitos a exibir:", 
                    5, 20, 10, 1, 
                    key="slider_top_temporal"
                )
                
                totais = {c: sum(anos.values()) for c, anos in conceito_ano.items()}
                top_conceitos = sorted(totais.items(), key=lambda x: x[1], reverse=True)[:top_n_temporal]
                
                # Determinar range de anos
                todos_anos = set()
                for c, _ in top_conceitos:
                    todos_anos.update(conceito_ano[c].keys())
                
                if todos_anos:
                    ano_min, ano_max = min(todos_anos), max(todos_anos)
                    anos_range = list(range(ano_min, ano_max + 1))
                    
                    # Criar figura
                    fig_temporal = go.Figure()
                    
                    # Paleta de cores distintas
                    cores = [
                        '#E41A1C', '#FF7F00', '#C4A000', '#4DAF4A', '#377EB8',
                        '#00CED1', '#984EA3', '#FF69B4', '#A65628', '#F781BF',
                        '#999999', '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3',
                        '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3', '#1B9E77'
                    ]
                    
                    for i, (conceito, total) in enumerate(top_conceitos):
                        # Frequências por ano (0 se não existir)
                        freqs = [conceito_ano[conceito].get(ano, 0) for ano in anos_range]
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=anos_range,
                            y=freqs,
                            mode='lines+markers',
                            name=f'{conceito} ({total})',
                            line=dict(color=cores[i % len(cores)], width=2),
                            marker=dict(size=6),
                            hovertemplate=f'<b>{conceito}</b><br>Ano: %{{x}}<br>Frequência: %{{y}}<extra></extra>'
                        ))
                    
                    fig_temporal.update_layout(
                        title=f'Top {top_n_temporal} Conceitos ao Longo do Tempo',
                        xaxis_title='Ano',
                        yaxis_title='Frequência',
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            title='Conceitos (Total)',
                            orientation='v',
                            yanchor='top',
                            y=1,
                            xanchor='left',
                            x=1.02,
                            font=dict(size=10)
                        ),
                        margin=dict(r=250),  # Espaço para legenda
                        xaxis=dict(
                            tickmode='linear',
                            dtick=1 if len(anos_range) <= 15 else 2
                        )
                    )
                    
                    st.plotly_chart(fig_temporal, width='stretch')
                    
                    # Tabela opcional
                    with st.expander("📋 Ver dados da evolução temporal"):
                        # Criar DataFrame pivot
                        dados_pivot = []
                        for conceito, total in top_conceitos:
                            row = {'Conceito': conceito, 'Total': total}
                            for ano in anos_range:
                                row[str(ano)] = conceito_ano[conceito].get(ano, 0)
                            dados_pivot.append(row)
                        
                        df_temporal = pd.DataFrame(dados_pivot)
                        st.dataframe(df_temporal, width='stretch')
                else:
                    st.info("Dados temporais insuficientes para gerar o gráfico.")
            else:
                st.info("Não foi possível extrair dados de conceitos por ano.")

        # ========== SUB-ABA 3: COOCORRÊNCIAS ==========
        with t3:
            st.header("🔗 Coocorrências")

            # Calcular pares
            pairs = Counter()
            for concepts in concepts_lists:
                for i, c1 in enumerate(concepts):
                    for c2 in concepts[i+1:]:
                        if c1 != c2:
                            pairs[tuple(sorted([c1, c2]))] += 1

            st.metric("Pares Únicos", len(pairs))

            st.divider()

            # Top pares
            st.subheader("🏆 Coocorrências Mais Frequentes")

            top_pairs = st.slider("Número de pares:", 10, 100, 30, 10, key="top_pairs")

            # Calcular frequências individuais para Salton
            concept_freq = dict(freq)
            
            df_pairs = pd.DataFrame([
                {
                    'Conceito 1': c1,
                    'Conceito 2': c2,
                    'Frequência': f,
                    'Salton': round(f / np.sqrt(concept_freq.get(c1, 1) * concept_freq.get(c2, 1)), 4)
                }
                for (c1, c2), f in pairs.most_common(top_pairs)
            ])

            st.dataframe(df_pairs, width="stretch")

            st.divider()

            # Matriz de calor
            st.subheader("🔥 Matriz de Calor de Coocorrências")

            top_heatmap = st.slider("Conceitos na matriz:", 5, 20, 10, 1, key="heatmap_size")

            top_concepts = [c for c, _ in freq.most_common(top_heatmap)]

            # Criar matriz
            matrix = pd.DataFrame(0, index=top_concepts, columns=top_concepts)

            for (c1, c2), f in pairs.items():
                if c1 in top_concepts and c2 in top_concepts:
                    matrix.loc[c1, c2] = f
                    matrix.loc[c2, c1] = f

            fig = px.imshow(
                matrix,
                labels=dict(x="Conceito", y="Conceito", color="Coocorrências"),
                title=f"Matriz de Calor - Top {top_heatmap} Conceitos",
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=600)

            st.plotly_chart(fig, width="stretch")

            st.divider()

            # Matriz de Similaridade de Salton
            st.subheader("📐 Matriz de Similaridade (Cosseno de Salton)")
            st.caption("Salton(i,j) = coocorrência(i,j) / √(freq(i) × freq(j)) — normaliza coocorrências, valores de 0 a 1")
            
            top_salton = st.slider("Conceitos na matriz de Salton:", 5, 20, 15, 1, key="salton_size")
            
            top_concepts_salton = [c for c, _ in freq.most_common(top_salton)]
            
            # Criar matriz de Salton
            salton_matrix = pd.DataFrame(0.0, index=top_concepts_salton, columns=top_concepts_salton)
            
            for (c1, c2), f in pairs.items():
                if c1 in top_concepts_salton and c2 in top_concepts_salton:
                    salton_value = f / np.sqrt(concept_freq.get(c1, 1) * concept_freq.get(c2, 1))
                    salton_matrix.loc[c1, c2] = round(salton_value, 4)
                    salton_matrix.loc[c2, c1] = round(salton_value, 4)
            
            fig_salton = px.imshow(
                salton_matrix,
                labels=dict(x="Conceito", y="Conceito", color="Similaridade"),
                title=f"Similaridade de Salton - Top {top_salton} Conceitos",
                color_continuous_scale='Greens'
            )
            fig_salton.update_layout(height=600)
            
            st.plotly_chart(fig_salton, width="stretch")
            
            # Botão para baixar matriz completa
            with st.expander("💾 Baixar Matriz Completa de Salton"):
                st.caption("Matriz com todos os conceitos do grafo")
                
                if 'cache_salton_csv' not in st.session_state:
                    all_concepts = list(freq.keys())
                    full_salton = pd.DataFrame(0.0, index=all_concepts, columns=all_concepts)
                    
                    for (c1, c2), f in pairs.items():
                        salton_value = f / np.sqrt(concept_freq.get(c1, 1) * concept_freq.get(c2, 1))
                        full_salton.loc[c1, c2] = round(salton_value, 4)
                        full_salton.loc[c2, c1] = round(salton_value, 4)
                    
                    st.session_state.cache_salton_csv = full_salton.to_csv()
                    st.session_state.cache_salton_dim = len(all_concepts)
                
                st.download_button(
                    "📥 Baixar Matriz Salton (CSV)",
                    data=st.session_state.cache_salton_csv,
                    file_name="matriz_salton_completa.csv",
                    mime="text/csv",
                    width="stretch",
                    key="dl_salton_csv"
                )
                
                st.metric("Dimensão da matriz", f"{len(all_concepts)} x {len(all_concepts)}")
            
            # Distribuição de frequências
            st.subheader("📈 Distribuição das Frequências de Coocorrência")

            freqs = list(pairs.values())

            fig3 = px.histogram(
                x=freqs,
                nbins=30,
                labels={'x': 'Frequência de coocorrência', 'y': 'Número de pares'},
                title="Distribuição das Frequências"
            )

            st.plotly_chart(fig3, width="stretch")

            # ========== EVOLUÇÃO TEMPORAL DAS COOCORRÊNCIAS ==========
            st.divider()
            st.subheader("📈 Evolução Temporal das Coocorrências")
            
            # Extrair pares de coocorrência por ano
            pares_por_ano = {}  # {ano: {(conceito1, conceito2): frequência}}
            
            for article in articles:
                ano = article.get('year')
                if not ano:
                    continue
                
                concepts = article.get('concepts', [])
                # Filtrar conceitos relevantes
                nomes = [
                    c.get('display_name', c.get('name', ''))
                    for c in concepts
                    if c.get('score', 0) >= 0.35 and (c.get('display_name') or c.get('name'))
                ]
                
                if len(nomes) >= 2:
                    if ano not in pares_por_ano:
                        pares_por_ano[ano] = {}
                    
                    # Gerar pares (ordem alfabética para consistência)
                    for i in range(len(nomes)):
                        for j in range(i + 1, len(nomes)):
                            par = tuple(sorted([nomes[i], nomes[j]]))
                            if par not in pares_por_ano[ano]:
                                pares_por_ano[ano][par] = 0
                            pares_por_ano[ano][par] += 1
            
            if pares_por_ano:
                # Calcular top pares globais
                totais_pares = {}
                for ano, pares in pares_por_ano.items():
                    for par, freq in pares.items():
                        if par not in totais_pares:
                            totais_pares[par] = 0
                        totais_pares[par] += freq
                
                # Selecionar top N pares
                top_n_pares = st.slider(
                    "Número de pares a exibir:",
                    5, 20, 10, 1,
                    key="slider_top_pares_temporal"
                )
                
                top_pares = sorted(totais_pares.items(), key=lambda x: x[1], reverse=True)[:top_n_pares]
                top_pares_set = {par for par, _ in top_pares}
                
                # Preparar dados para visualização
                anos_ordenados = sorted(pares_por_ano.keys())
                
                # ===== HEATMAP TEMPORAL =====
                st.markdown("#### 🔥 Mapa de Calor: Coocorrências × Anos")
                
                # Criar matriz para heatmap
                pares_labels = [f"{p[0]} ↔ {p[1]}" for p, _ in top_pares]
                matriz_heatmap = []
                
                for par, _ in top_pares:
                    row = []
                    for ano in anos_ordenados:
                        freq_valor = pares_por_ano.get(ano, {}).get(par, 0)
                        row.append(freq_valor)
                    matriz_heatmap.append(row)
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=matriz_heatmap,
                    x=[str(a) for a in anos_ordenados],
                    y=pares_labels,
                    colorscale='Blues',
                    hovertemplate='Par: %{y}<br>Ano: %{x}<br>Frequência: %{z}<extra></extra>'
                ))
                
                fig_heatmap.update_layout(
                    title='Intensidade de Coocorrência ao Longo do Tempo',
                    xaxis_title='Ano',
                    yaxis_title='Par de Conceitos',
                    height=max(400, top_n_pares * 35),
                    yaxis=dict(tickfont=dict(size=10)),
                    xaxis=dict(tickangle=45)
                )
                
                st.plotly_chart(fig_heatmap, width='stretch')
                
                # ===== SANKEY DE EVOLUÇÃO TEMÁTICA =====
                st.markdown("#### 🌊 Diagrama Sankey: Fluxo Temporal das Coocorrências")
                
                # Dividir em períodos (automático baseado nos dados)
                if len(anos_ordenados) >= 4:
                    # Criar 3 períodos
                    n_anos = len(anos_ordenados)
                    corte1 = n_anos // 3
                    corte2 = 2 * n_anos // 3
                    
                    periodos = {
                        f"{anos_ordenados[0]}-{anos_ordenados[corte1-1]}": anos_ordenados[:corte1],
                        f"{anos_ordenados[corte1]}-{anos_ordenados[corte2-1]}": anos_ordenados[corte1:corte2],
                        f"{anos_ordenados[corte2]}-{anos_ordenados[-1]}": anos_ordenados[corte2:]
                    }
                    
                    # Calcular top pares por período
                    top_por_periodo = {}
                    for periodo, anos_periodo in periodos.items():
                        freq_periodo = {}
                        for ano in anos_periodo:
                            for par, freq in pares_por_ano.get(ano, {}).items():
                                if par not in freq_periodo:
                                    freq_periodo[par] = 0
                                freq_periodo[par] += freq
                        
                        top_por_periodo[periodo] = sorted(
                            freq_periodo.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:7]  # Top 7 por período
                    
                    # Construir Sankey
                    nodes = []
                    node_indices = {}
                    
                    # Criar nós para cada período
                    periodos_lista = list(periodos.keys())
                    for periodo in periodos_lista:
                        for par, freq in top_por_periodo[periodo]:
                            node_label = f"{par[0][:12]}—{par[1][:12]}"
                            if node_label not in node_indices:
                                node_indices[node_label] = len(nodes)
                                nodes.append({
                                    'label': node_label,
                                    'par': par,
                                    'periodo': periodo,
                                    'freq': freq
                                })
                    
                    # Criar links entre períodos
                    links_source = []
                    links_target = []
                    links_value = []
                    
                    for i in range(len(periodos_lista) - 1):
                        periodo_atual = periodos_lista[i]
                        periodo_prox = periodos_lista[i + 1]
                        
                        for par_atual, freq_atual in top_por_periodo[periodo_atual]:
                            for par_prox, freq_prox in top_por_periodo[periodo_prox]:
                                # Verificar sobreposição de conceitos
                                set_atual = set(par_atual)
                                set_prox = set(par_prox)
                                overlap = len(set_atual & set_prox)
                                
                                if overlap >= 1:  # Pelo menos um conceito em comum
                                    node_label_atual = f"{par_atual[0][:12]}—{par_atual[1][:12]}"
                                    node_label_prox = f"{par_prox[0][:12]}—{par_prox[1][:12]}"
                                    
                                    if node_label_atual in node_indices and node_label_prox in node_indices:
                                        links_source.append(node_indices[node_label_atual])
                                        links_target.append(node_indices[node_label_prox])
                                        # Valor proporcional à força da conexão
                                        links_value.append(min(freq_atual, freq_prox) * overlap)
                    
                    if links_source:
                        # Cores por período
                        cores_periodo = {
                            periodos_lista[0]: 'rgba(31, 119, 180, 0.8)',
                            periodos_lista[1]: 'rgba(255, 127, 14, 0.8)',
                            periodos_lista[2]: 'rgba(44, 160, 44, 0.8)'
                        }
                        
                        node_colors = [
                            cores_periodo.get(n['periodo'], 'rgba(150,150,150,0.8)')
                            for n in nodes
                        ]
                        
                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=20,
                                thickness=25,
                                line=dict(color="white", width=1),
                                label=[n['label'] for n in nodes],
                                color=node_colors,
                                hovertemplate='%{label}<br>Frequência: %{value}<extra></extra>'
                            ),
                            link=dict(
                                source=links_source,
                                target=links_target,
                                value=links_value,
                                color='rgba(180,180,180,0.5)'
                            ),
                            textfont=dict(
                                family='Arial',
                                size=11,
                                color='#333333'
                            )
                        )])
                        
                        fig_sankey.update_layout(
                            title='Evolução Temática das Coocorrências entre Períodos',
                            height=600,
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_sankey, width='stretch')
                        
                        # Legenda
                        st.markdown(f"""
                        **Legenda dos Períodos:**
                        - 🔵 {periodos_lista[0]} (período inicial)
                        - 🟠 {periodos_lista[1]} (período intermediário)  
                        - 🟢 {periodos_lista[2]} (período recente)
                        
                        *As conexões indicam continuidade temática (conceitos compartilhados entre pares).*
                        """)
                    else:
                        st.info("Dados insuficientes para gerar o diagrama Sankey.")
                else:
                    st.info("São necessários pelo menos 4 anos de dados para gerar o diagrama de evolução.")
                
                # Tabela expandível
                with st.expander("📋 Ver dados das coocorrências temporais"):
                    dados_tabela = []
                    for par, total in top_pares:
                        row = {
                            'Par': f"{par[0]} ↔ {par[1]}",
                            'Total': total
                        }
                        for ano in anos_ordenados:
                            row[str(ano)] = pares_por_ano.get(ano, {}).get(par, 0)
                        dados_tabela.append(row)
                    
                    df_pares_temporal = pd.DataFrame(dados_tabela)
                    st.dataframe(df_pares_temporal, width='stretch')
            else:
                st.info("Não foi possível extrair dados de coocorrência temporal.")

        # ========== SUB-ABA 4: GRAFO ==========
        with t4:
            st.header("🕸️ Grafo")

            # Métricas do grafo
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Nós", len(G.nodes()))
            col2.metric("Arestas", len(G.edges()))

            if len(G.nodes()) > 0:
                col3.metric("Densidade", f"{nx.density(G):.4f}")
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                col4.metric("Grau Médio", f"{avg_degree:.2f}")

            if len(G.nodes()) > 0:
                st.divider()

                # Centralidade
                st.subheader("📊 Análise de Centralidade")

                tipo_centralidade = st.selectbox(
                    "Tipo de centralidade:",
                    ["Grau", "Intermediação", "Proximidade"],
                    key="centrality_type"
                )

                if tipo_centralidade == "Grau":
                    centrality = nx.degree_centrality(G)
                elif tipo_centralidade == "Intermediação":
                    centrality = nx.betweenness_centrality(G)
                else:
                    centrality = nx.closeness_centrality(G)

                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]

                df_central = pd.DataFrame(top_central, columns=['Conceito', 'Centralidade'])

                fig = px.bar(
                    df_central,
                    x='Centralidade',
                    y='Conceito',
                    orientation='h',
                    title=f"Top 20 - Centralidade de {tipo_centralidade}",
                    color='Centralidade',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig, width="stretch")

                st.divider()

                # Comunidades
                st.subheader("👥 Detecção de Comunidades (Cluster)")

                try:
                    from networkx.algorithms import community
                    communities = list(community.greedy_modularity_communities(G))

                    st.metric("Número de Comunidades", len(communities))

                    for i, comm in enumerate(communities, 1):
                        with st.expander(f"Comunidade {i} ({len(comm)} conceitos)"):
                            members = sorted(list(comm))
                            st.write(", ".join(members))

                except Exception as e:
                    st.info("Não foi possível detectar comunidades")

                st.divider()

                # Visualização interativa
                st.subheader("🎨 Visualização Estática")

                if len(G.nodes()) <= 100:
                    top_viz = st.slider("Nós a visualizar:", 5, min(50, len(G.nodes())), 15, key="viz_nodes")

                    top_nodes = [n for n, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_viz]]

                    Gv = G.subgraph(top_nodes).copy()
                    pos = nx.spring_layout(Gv, k=0.5, iterations=50, seed=42)

                    # Criar traços
                    edge_trace = go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none'
                    )

                    for edge in Gv.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace['x'] += tuple([x0, x1, None])
                        edge_trace['y'] += tuple([y0, y1, None])

                    node_trace = go.Scatter(
                        x=[],
                        y=[],
                        mode='markers+text',
                        hoverinfo='text',
                        text=[],
                        textposition="top center",
                        marker=dict(
                            size=[],
                            color=[],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Centralidade")
                        )
                    )

                    for node in Gv.nodes():
                        x, y = pos[node]
                        node_trace['x'] += tuple([x])
                        node_trace['y'] += tuple([y])
                        node_trace['text'] += tuple([node[:20]])
                        node_trace['marker']['size'] += tuple([centrality[node] * 50 + 10])
                        node_trace['marker']['color'] += tuple([centrality[node]])

                    fig = go.Figure(
                        data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Rede Interativa de Conceitos",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=700
                        )
                    )

                    st.plotly_chart(fig, width="stretch")

                else:
                    st.warning("⚠️ Grafo muito grande (>100 nós). Use filtros para reduzir o tamanho.")

        # ========== SUB-ABA 5: MAPA TEMÁTICO =========
        with t5:
            st.header("🗺️ Mapa Temático")

            st.markdown("""
            O **Mapa Temático** organiza os conceitos em clusters e os classifica em quatro quadrantes
            a partir de centralidade (importância no campo) e densidade (coesão interna):

            - 🎯 **Temas Motores**: Centrais e bem desenvolvidos (PRIORIZE)
            - 🔷 **Temas Nicho**: Especializados e coesos
            - 🔶 **Temas Básicos**: Transversais, mas em desenvolvimento
            - 🔴 **Temas Emergentes / Declinantes**: Fronteiras de pesquisa
            """)

            if len(G.nodes()) < 5:
                st.warning("⚠️ Poucos conceitos no grafo para gerar um mapa temático confiável (mínimo ≈ 10).")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    cluster_method = st.selectbox(
                        "Método de Clusterização:",
                        ["louvain", "greedy"],
                        help="Algoritmo para detectar comunidades no grafo de coocorrências"
                    )

                with col2:
                    min_cluster_size = st.slider(
                        "Tamanho mínimo do cluster:",
                        min_value=2,
                        max_value=10,
                        value=3,
                        help="Quantidade mínima de conceitos por cluster"
                    )

                if st.button("🔍 Gerar Mapa Temático", type="primary", key="generate_thematic_map"):
                    try:
                        # Chama a função cacheada definida no topo
                        with st.spinner("🔄 Detectando clusters e calculando métricas do mapa temático..."):
                            # Recebemos o DataFrame (metrics_df) e a lista de clusters (clusters_detected)
                            metrics_df, clusters_detected = run_cached_thematic_map(G, concepts_lists, cluster_method, min_cluster_size)

                        if metrics_df is None or len(metrics_df) == 0:
                            st.warning("⚠️ Nenhum cluster detectado. Verifique os parâmetros ou amplie o corpus.")
                        else:
                            # ---------- Converter métricas em estrutura 'thematic_data' ----------
                            thematic_data = []
                            tipo_map = {
                                "Q1: Motor Themes": "Motor Theme",
                                "Q2: Basic/Transversal Themes": "Basic Theme",
                                "Q3: Niche Themes": "Niche Theme",
                                "Q4: Emerging/Declining Themes": "Emerging/Declining Theme",
                            }

                            centralidades = []
                            densidades = []

                            # garante alinhamento: mesma ordem de metrics_df e clusters_detected
                            for idx, row in metrics_df.reset_index(drop=True).iterrows():
                                quadrante = row.get("quadrante", "Basic Theme") # Fallback seguro
                                if "quadrante" not in row:
                                    # Recalcula quadrante se não vier do cache (segurança)
                                    from thematic_map_module import ThematicMapAnalyzer
                                    quadrante = ThematicMapAnalyzer.classify_quadrant(
                                        row["centralidade_norm"], 
                                        row["densidade_norm"]
                                    )
                                
                                tipo = tipo_map.get(quadrante, "Basic Theme")

                                # conceitos do cluster (set -> lista ordenada)
                                conceitos_cluster = sorted(clusters_detected[idx])
                                tamanho_cluster = len(conceitos_cluster)

                                # conceito principal: primeiro da lista de principais ou primeiro do cluster
                                if isinstance(row.get("conceitos_principais", ""), str) and row["conceitos_principais"].strip():
                                    conceito_principal = row["conceitos_principais"].split(",")[0].strip()
                                else:
                                    conceito_principal = conceitos_cluster[0] if conceitos_cluster else ""

                                registro = {
                                    "cluster_id": idx + 1,
                                    "nome": row["nome"],
                                    "tipo": tipo,
                                    "tamanho": tamanho_cluster,
                                    "centralidade": float(row["centralidade"]),
                                    "densidade": float(row["densidade"]),
                                    "conceitos": conceitos_cluster,
                                    "conceito_principal": conceito_principal,
                                }

                                thematic_data.append(registro)
                                centralidades.append(registro["centralidade"])
                                densidades.append(registro["densidade"])

                            if not thematic_data:
                                st.warning("⚠️ Clusters foram detectados, mas não foi possível montar o mapa temático.")
                            else:
                                # ---------- Métricas de topo ----------
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total de Clusters", len(thematic_data))
                                with col2:
                                    motor_themes = sum(1 for t in thematic_data if t["tipo"] == "Motor Theme")
                                    st.metric("Motor Themes", motor_themes)
                                with col3:
                                    total_concepts = sum(t["tamanho"] for t in thematic_data)
                                    st.metric("Conceitos Agrupados", total_concepts)
                                with col4:
                                    st.metric("Tamanho Médio", f"{total_concepts / len(thematic_data):.1f}")

                                # ---------- Diagrama estratégico ----------
                                med_centrality = sum(centralidades) / len(centralidades)
                                med_density = sum(densidades) / len(densidades)

                                color_map = {
                                    "Motor Theme": "#2ecc71",               # verde
                                    "Niche Theme": "#3498db",               # azul
                                    "Emerging/Declining Theme": "#e74c3c",  # vermelho
                                    "Basic Theme": "#f39c12",               # amarelo
                                }

                                colors_list = [color_map.get(t["tipo"], "#95a5a6") for t in thematic_data]

                                fig_mapa = go.Figure()

                                fig_mapa.add_hline(y=med_density, line_dash="dash", line_color="gray")
                                fig_mapa.add_vline(x=med_centrality, line_dash="dash", line_color="gray")

                                fig_mapa.add_trace(go.Scatter(
                                    x=centralidades,
                                    y=densidades,
                                    mode="markers+text",
                                    marker=dict(
                                        size=[20 + t["tamanho"] * 5 for t in thematic_data],
                                        color=colors_list,
                                        line=dict(width=2, color="white"),
                                        opacity=0.85,
                                    ),
                                    text=[f"C{t['cluster_id']}" for t in thematic_data],
                                    textposition="middle center",
                                    textfont=dict(size=10, color="white", family="Arial Black"),
                                    hovertemplate=(
                                        "<b>%{customdata[0]}</b><br>" +
                                        "Centralidade: %{x:.3f}<br>" +
                                        "Densidade: %{y:.3f}<br>" +
                                        "Tipo: %{customdata[1]}<br>" +
                                        "Tamanho: %{customdata[2]} conceitos<br>" +
                                        "<extra></extra>"
                                    ),
                                    customdata=[
                                        [t["nome"], t["tipo"], t["tamanho"]]
                                        for t in thematic_data
                                    ],
                                    showlegend=False
                                ))

                                fig_mapa.update_layout(
                                    title="Diagrama Estratégico dos Clusters Temáticos",
                                    xaxis_title="Centralidade",
                                    yaxis_title="Densidade",
                                    height=600,
                                    plot_bgcolor="white",
                                    xaxis=dict(gridcolor="lightgray"),
                                    yaxis=dict(gridcolor="lightgray"),
                                )

                                annot_font = dict(size=13, family="Arial Black")

                                # Q1: Motor (Topo Direito)
                                fig_mapa.add_annotation(
                                    x=0.98, y=0.98, xref="paper", yref="paper",
                                    text="<b>MOTOR THEMES</b><br><span style='font-size:10px; font-weight:normal'>(Central + Desenvolvido)</span>",
                                    showarrow=False, xanchor="right", yanchor="top",
                                    font=dict(color="#2ecc71", **annot_font),
                                    bgcolor="rgba(255,255,255,0.7)"
                                )
                                # Q2: Basic (Base Direita)
                                fig_mapa.add_annotation(
                                    x=0.98, y=0.02, xref="paper", yref="paper",
                                    text="<b>BASIC THEMES</b><br><span style='font-size:10px; font-weight:normal'>(Central + Não-desenvolvido)</span>",
                                    showarrow=False, xanchor="right", yanchor="bottom",
                                    font=dict(color="#f39c12", **annot_font),
                                    bgcolor="rgba(255,255,255,0.7)"
                                )
                                # Q3: Niche (Topo Esquerdo)
                                fig_mapa.add_annotation(
                                    x=0.02, y=0.98, xref="paper", yref="paper",
                                    text="<b>NICHE THEMES</b><br><span style='font-size:10px; font-weight:normal'>(Periférico + Desenvolvido)</span>",
                                    showarrow=False, xanchor="left", yanchor="top",
                                    font=dict(color="#3498db", **annot_font),
                                    bgcolor="rgba(255,255,255,0.7)"
                                )
                                # Q4: Emerging (Base Esquerda)
                                fig_mapa.add_annotation(
                                    x=0.02, y=0.02, xref="paper", yref="paper",
                                    text="<b>EMERGING/DECLINING</b><br><span style='font-size:10px; font-weight:normal'>(Periférico + Não-desenvolvido)</span>",
                                    showarrow=False, xanchor="left", yanchor="bottom",
                                    font=dict(color="#e74c3c", **annot_font),
                                    bgcolor="rgba(255,255,255,0.7)"
                                )

                                st.plotly_chart(fig_mapa, width="stretch")

                                # ---------- Detalhamento dos clusters ----------
                                st.markdown("### 📋 Detalhamento dos Clusters")

                                tipo_icons = {
                                    "Motor Theme": "🎯",
                                    "Basic Theme": "🔶",
                                    "Niche Theme": "💎",
                                    "Emerging/Declining Theme": "🔴",
                                }

                                for cluster in thematic_data:
                                    icon = tipo_icons.get(cluster["tipo"], "⚪")

                                    with st.expander(f"{icon} {cluster['nome']} - {cluster['tipo']}"):
                                        col1, col2 = st.columns([2, 1])

                                        with col1:
                                            st.write("**Conceitos:**")
                                            concepts_display = ", ".join(sorted(cluster["conceitos"]))
                                            st.write(concepts_display)

                                        with col2:
                                            st.metric("Centralidade", f"{cluster['centralidade']:.3f}")
                                            st.metric("Densidade", f"{cluster['densidade']:.3f}")
                                            st.metric("Tamanho", cluster["tamanho"])

                                        # Interpretação sintética
                                        if "Motor" in cluster["tipo"]:
                                            st.success("🎯 Tema central e maduro. **PRIORIZE** na revisão de literatura.")
                                        elif "Niche" in cluster["tipo"]:
                                            st.info(f"💎 Tema especializado. Útil para nichos relacionados a '{cluster['conceito_principal']}'.")
                                        elif "Basic" in cluster["tipo"]:
                                            st.warning("🔶 Tema transversal. Oportunidade para pesquisas integradoras.")
                                        else:
                                            st.error("🔴 Tema emergente ou em declínio. Fronteira de pesquisa.")

                            # ---------- Explicação metodológica ----------
                            with st.expander("ℹ️ Sobre a metodologia"):
                                st.markdown("""
                                Este mapa temático segue a lógica do *Diagrama Estratégico*:

                                - **Densidade**: média dos pesos das arestas internas do cluster (coesão interna).
                                - **Centralidade**: soma dos pesos das arestas que ligam o cluster a outros clusters (relevância global).
                                - A posição de cada cluster no plano Centralidade × Densidade permite interpretar seu papel
                                  na estrutura do campo de pesquisa.

                                Referências:

                                - ARIA, M.; CUCCURULLO, C. Bibliometrix: An R-tool for comprehensive science mapping analysis. *Journal of Informetrics*, v.11, n.4, p.959–975, 2017. Doi: http://dx.doi.org/10.1016/j.joi.2017.08.007
                                - HE, Q. (1999). Knowledge discovery through co-word analysis. *Library Trends*, v.48, n.1, p.133–159, 1999. Disponível em: https://www.proquest.com/scholarly-journals/knowledge-discovery-through-co-word-analysis/docview/220452924/se-2 
                                """)

                    except ImportError:
                        st.error("""
                        Não foi possível importar o módulo `thematic_map_module`.
                        Verifique se o arquivo `thematic_map_module.py` está no mesmo diretório
                        de `streamlit_app.py` e se você executou a célula que o cria no Colab.
                        """)
                    except Exception as e:
                        st.error(f"Erro ao gerar mapa temático: {e}")
                        with st.expander("🐛 Detalhes técnicos do erro"):
                            st.exception(e)

        # ========== SUB-ABA 6: ESTATÍSTICAS ==========
        with t6:
            st.header("📊 Estatísticas")

            st.subheader("📋 Resumo Geral")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📚 Dados:**")
                st.write(f"• Artigos: {len(articles)}")
                st.write(f"• Com conceitos: {len(concepts_lists)}")
                if len(articles) > 0:
                    st.write(f"• Aproveitamento: {len(concepts_lists)/len(articles)*100:.1f}%")
                st.write(f"• Conceitos total: {len(all_concepts)}")
                st.write(f"• Únicos: {len(set(all_concepts))}")

            with col2:
                st.markdown("**🕸️ Grafo:**")
                st.write(f"• Nós: {len(G.nodes())}")
                st.write(f"• Arestas: {len(G.edges())}")
                if len(G.nodes()) > 0:
                    st.write(f"• Densidade: {nx.density(G):.4f}")
                    if nx.is_connected(G):
                        st.write(f"• Diâmetro: {nx.diameter(G)}")
                    else:
                        st.write(f"• Diâmetro: N/A (grafo desconexo)")
                    st.write(f"• Componentes: {nx.number_connected_components(G)}")

            st.divider()

            # Distribuições
            st.subheader("📈 Distribuições")

            col1, col2 = st.columns(2)

            with col1:
                if len(G.nodes()) > 0:
                    degrees = [d for n, d in G.degree()]

                    fig = px.histogram(
                        x=degrees,
                        nbins=20,
                        labels={'x': 'Grau', 'y': 'Frequência'},
                        title="Distribuição de Graus"
                    )

                    st.plotly_chart(fig, width="stretch")

            with col2:
                if len(G.edges()) > 0:
                    weights = [d['weight'] for u, v, d in G.edges(data=True)]

                    fig = px.histogram(
                        x=weights,
                        nbins=20,
                        labels={'x': 'Peso', 'y': 'Frequência'},
                        title="Distribuição dos Pesos das Arestas"
                    )

                    st.plotly_chart(fig, width="stretch")

        # ========== SUB-ABA 7: EXPORTAÇÃO (ATUALIZADA) ==========
        with t7:
            st.header("💾 Exportação")

            col1, col2, col3 = st.columns(3)

            # --- COLUNA 1: JSON ---
            with col1:
                st.subheader("📄 JSON")

                if 'cache_artigos_json' not in st.session_state:
                    st.session_state.cache_artigos_json = json.dumps(articles, indent=2, ensure_ascii=False)

                st.download_button(
                    "📥 Artigos (JSON Completo)",
                    st.session_state.cache_artigos_json,
                    "articles.json",
                    "application/json",
                    help="Dados brutos completos (ideal para mineração).",
                    width="stretch",
                    key="dl_artigos_json"
                )

                if 'cache_conceitos_json' not in st.session_state:
                    st.session_state.cache_conceitos_json = json.dumps(concepts_lists, indent=2, ensure_ascii=False)

                st.download_button(
                    "📥 Conceitos (JSON)",
                    st.session_state.cache_conceitos_json,
                    "concepts.json",
                    "application/json",
                    width="stretch",
                    key="dl_conceitos_json"
                )

                if 'cache_cooc_json' not in st.session_state:
                    cooc_json = [
                        {"conceito1": c1, "conceito2": c2, "frequencia": f}
                        for (c1, c2), f in pairs.items()
                    ]
                    st.session_state.cache_cooc_json = json.dumps(cooc_json, indent=2, ensure_ascii=False)

                st.download_button(
                    "📥 Coocorrências (JSON)",
                    st.session_state.cache_cooc_json,
                    "cooccurrences.json",
                    "application/json",
                    width="stretch",
                    key="dl_cooc_json"
                )

            # --- COLUNA 2: CSV ---
            with col2:
                st.subheader("📊 CSV")

                if 'cache_artigos_csv' not in st.session_state:
                    df_articles_export = pd.DataFrame([
                        {
                            'title': a.get('title', ''),
                            'year': a.get('year', ''),
                            'num_concepts': len(a.get('concepts', []))
                        }
                        for a in articles
                    ])
                    st.session_state.cache_artigos_csv = df_articles_export.to_csv(index=False)

                st.download_button(
                    "📥 Artigos (CSV)",
                    st.session_state.cache_artigos_csv,
                    "articles.csv",
                    "text/csv",
                    width="stretch",
                    key="dl_artigos_csv"
                )

                if 'cache_conceitos_csv' not in st.session_state:
                    from collections import Counter
                    all_concepts_export = []
                    for a in articles:
                        for c in a.get('concepts', []):
                            nome = c.get('display_name', c.get('name', ''))
                            if nome:
                                all_concepts_export.append(nome)
                    freq_export = Counter(all_concepts_export)
                    
                    df_concepts = pd.DataFrame(
                        freq_export.most_common(),
                        columns=['concept', 'frequency']
                    )
                    st.session_state.cache_conceitos_csv = df_concepts.to_csv(index=False)

                st.download_button(
                    "📥 Conceitos (CSV)",
                    st.session_state.cache_conceitos_csv,
                    "concepts.csv",
                    "text/csv",
                    width="stretch",
                    key="dl_conceitos_csv"
                )

                if 'cache_cooc_csv' not in st.session_state:
                    edges_list = [[u, v, d['weight']] for u, v, d in G.edges(data=True)]
                    df_cooc = pd.DataFrame(edges_list, columns=['source', 'target', 'weight'])
                    st.session_state.cache_cooc_csv = df_cooc.to_csv(index=False)

                st.download_button(
                    "📥 Coocorrências (CSV)",
                    st.session_state.cache_cooc_csv,
                    "cooccurrences.csv",
                    "text/csv",
                    width="stretch",
                    key="dl_cooc_csv"
                )

            # --- COLUNA 3: OUTROS FORMATOS (GraphML e .net) ---
            with col3:
                st.subheader("🔧 Redes")

                # 1. GraphML
                try:
                    if 'cache_graphml_painel' not in st.session_state:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.graphml') as f:
                            nx.write_graphml(G, f.name)
                        with open(f.name, 'r', encoding='utf-8') as file:
                            st.session_state.cache_graphml_painel = file.read()
                    
                    st.download_button(
                        "📥 Grafo (GraphML)",
                        st.session_state.cache_graphml_painel,
                        "graph.graphml",
                        "application/xml",
                        help="Para Gephi ou Cytoscape",
                        width="stretch",
                        key="dl_graphml_painel"
                    )
                except Exception as e:
                    st.error(f"Erro GraphML: {e}")

                # 2. Pajek .net
                try:
                    if 'cache_pajek_painel' not in st.session_state:
                        st.session_state.cache_pajek_painel = exp.generate_pajek_net(G)
                    
                    st.download_button(
                        "📥 Grafo (.net Pajek)", 
                        st.session_state.cache_pajek_painel, 
                        "graph.net", 
                        "text/plain",
                        help="Para VOSviewer ou Pajek",
                        width="stretch",
                        key="dl_pajek_painel"
                    )
                except Exception as e:
                    st.error(f"Erro Pajek: {e}")

            st.divider()
                   
            # --- DADOS RICOS (BIBTEX, RIS, EXCEL) ---
            st.subheader("📤 Exportação de Dados e Referências")
            st.caption("Arquivos processados com metadados completos (DOI, Abstract, Keywords).")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            # 1. Botão Excel
            try:
                if 'cache_excel' not in st.session_state:
                    st.session_state.cache_excel = exp.generate_excel(articles)
                with col_exp1:
                    st.download_button(
                        label="📊 Excel (.xlsx)",
                        data=st.session_state.cache_excel,
                        file_name="delineia_resultados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Planilha formatada com conceitos, score e level.",
                        width="stretch",
                        key="dl_excel"
                    )
            except Exception as e:
                st.error(f"Erro Excel: {e}")
            
            # 2. Botão BibTeX
            try:
                if 'cache_bibtex' not in st.session_state:
                    st.session_state.cache_bibtex = exp.generate_bibtex(articles)
                with col_exp2:
                    st.download_button(
                        label="🎓 BibTeX (.bib)",
                        data=st.session_state.cache_bibtex,
                        file_name="delineia_referencias.bib",
                        mime="text/plain",
                        help="Para LaTeX/Overleaf.",
                        width="stretch",
                        key="dl_bibtex"
                    )
            except Exception as e:
                st.error(f"Erro BibTeX: {e}")
                
            # 3. Botão RIS
            try:
                if 'cache_ris' not in st.session_state:
                    st.session_state.cache_ris = exp.generate_ris(articles)
                with col_exp3:
                    st.download_button(
                        label="📚 RIS (Zotero)",
                        data=st.session_state.cache_ris,
                        file_name="delineia_referencias.ris",
                        mime="application/x-research-info-systems",
                        help="Para Zotero, Mendeley, EndNote.",
                        width="stretch",
                        key="dl_ris"
                    )
            except Exception as e:
                st.error(f"Erro RIS: {e}")

            # --- PACOTE ZIP ---
            st.subheader("📦 Pacote Completo")

            if st.button("🎁 Gerar ZIP com Todos os Dados", width="stretch", key="btn_gerar_zip"):
                with st.spinner("📦 Gerando arquivo ZIP..."):
                    import zipfile
                    from io import BytesIO
                    from datetime import datetime
                    
                    zip_buffer = BytesIO()

                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        # JSON (usa cache ou gera na hora)
                        zf.writestr('articles.json', st.session_state.get('cache_artigos_json', json.dumps(articles, indent=2, ensure_ascii=False)))
                        zf.writestr('concepts.json', st.session_state.get('cache_conceitos_json', json.dumps(concepts_lists, indent=2, ensure_ascii=False)))
                        cooc_json_zip = st.session_state.get('cache_cooc_json', json.dumps([{"conceito1": c1, "conceito2": c2, "frequencia": f} for (c1, c2), f in pairs.items()], indent=2, ensure_ascii=False))
                        zf.writestr('cooccurrences.json', cooc_json_zip)

                        # CSV (usa cache ou gera na hora)
                        zf.writestr('articles.csv', st.session_state.get('cache_artigos_csv', pd.DataFrame([{'title': a.get('title', ''), 'year': a.get('year', ''), 'num_concepts': len(a.get('concepts', []))} for a in articles]).to_csv(index=False)))
                        zf.writestr('concepts.csv', st.session_state.get('cache_conceitos_csv', ''))
                        zf.writestr('cooccurrences.csv', st.session_state.get('cache_cooc_csv', pd.DataFrame([[u, v, d['weight']] for u, v, d in G.edges(data=True)], columns=['source', 'target', 'weight']).to_csv(index=False)))

                        # Redes (usa cache ou gera na hora)
                        graphml_zip = st.session_state.get('cache_graphml_painel', '')
                        if graphml_zip:
                            zf.writestr('graph.graphml', graphml_zip)
                        
                        # Pajek .net
                        try:
                            pajek_zip = st.session_state.get('cache_pajek_painel', exp.generate_pajek_net(G))
                            if isinstance(pajek_zip, bytes):
                                zf.writestr('graph.net', pajek_zip)
                            else:
                                zf.writestr('graph.net', pajek_zip.encode('utf-8'))
                        except:
                            pass  # Se falhar o pajek, gera o zip sem ele

                        # --- DADOS RICOS (Excel, BibTeX, RIS) ---
                        # Excel (.xlsx)
                        try:
                            excel_bytes = exp.generate_excel(articles)
                            zf.writestr('delineia_dados.xlsx', excel_bytes)
                        except Exception as e:
                            print(f"Erro ao incluir Excel no ZIP: {e}")

                        # BibTeX (.bib)
                        try:
                            bib_str = exp.generate_bibtex(articles)
                            zf.writestr('delineia_referencias.bib', bib_str)
                        except Exception as e:
                            print(f"Erro ao incluir BibTeX no ZIP: {e}")

                        # RIS (.ris)
                        try:
                            ris_str = exp.generate_ris(articles)
                            zf.writestr('delineia_referencias.ris', ris_str)
                        except Exception as e:
                            print(f"Erro ao incluir RIS no ZIP: {e}")

                        # README ATUALIZADO
                        readme = f"""# Delinéia - Dados Exportados
Data: {datetime.now().strftime("%d/%m/%Y às %H:%M")}

Arquivos no pacote:
1. DADOS COMPLETOS (Para leitura humana e importação)
   - delineia_dados.xlsx: Excel com metadados, autores e conceitos
   - delineia_referencias.bib: Para LaTeX/Overleaf
   - delineia_referencias.ris: Para Zotero/Mendeley

2. DADOS BRUTOS (Para mineração)
   - articles.json / csv
   - concepts.json / csv
   - cooccurrences.json / csv

3. REDES (Para visualização)
   - graph.graphml (Gephi)
   - graph.net (Pajek/VOSviewer)

Total de Artigos: {len(articles)}
"""
                        zf.writestr('README.txt', readme)

                    st.download_button(
                        "📥 Baixar painel_completo.zip",
                        zip_buffer.getvalue(),
                        "painel_completo.zip",
                        "application/zip",
                        width="stretch",
                        key="dl_zip_completo"
                    )

    else:
    # O que mostrar se não tiver dados
        st.info("☝️ Realize uma busca para visualizar o painel.")
        # Mostrar exemplo
        with st.expander("💡 Exemplo de uso", expanded=True):
            st.markdown("""
            **Como usar o Painel:**
                        
            1. **Digite uma chave de busca** (ex: "machine learning" AND education)
            2. **Ajuste os filtros** conforme necessário
            3. **Clique em Buscar** para processar
            4. **Explore as abas** com diferentes análises
            5. **Exporte os dados** quando necessário
            
            **Dica:** Você pode copiar as chaves de busca do Delineascópio ou do Construtor de Chaves!
            """)

    rodape_institucional()