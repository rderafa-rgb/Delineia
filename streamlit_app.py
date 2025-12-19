# -*- coding: utf-8 -*-

import streamlit as st
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
import io
import tempfile
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
import gc

def limpar_memoria():
    """Força coleta de lixo"""
    gc.collect()

# ==================== FUNÇÕES COM CACHE (OTIMIZAÇÃO DE MEMÓRIA) ====================

@st.cache_resource
def get_pipeline_instance():
    """Cache da instância do pipeline para não recriar objetos pesados."""
    return ResearchScopePipeline(OPENALEX_EMAIL)

@st.cache_data(ttl="2h", show_spinner=False)
def run_cached_pipeline(nome, tema, questao, kws, genero):
    """
    Cache do processamento pesado. 
    Se os inputs forem os mesmos, retorna o resultado da RAM sem reprocessar.
    TTL de 2h limpa a memória automaticamente após inatividade.
    """
    pipe = get_pipeline_instance()
    # A função process retorna dicionários e grafos NetworkX, que o Streamlit serializa bem
    return pipe.process(nome, tema, questao, kws, genero=genero)

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

@st.cache_data(show_spinner=False)
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
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
<div style="display: flex; gap: 40px; align-items: center; margin-bottom: 15px;">
<img src="data:image/png;base64,{img_ufrgs}" width="160" style="opacity: 0.9;">
<img src="data:image/png;base64,{img_cinted}" width="160" style="opacity: 0.9;">
<img src="data:image/png;base64,{img_ppgie}" width="160" style="opacity: 0.9;">
</div>
<div style="text-align: center; color: #888888; font-size: 0.85rem; line-height: 1.5;">
<b>Delinéia</b> - Sistema de Apoio ao Delineamento de Escopo Temático<br>
Pesquisa de Doutorado - PPGIE / UFRGS<br>
2025
</div>
</div>
"""
    st.markdown(html_code, unsafe_allow_html=True)

# ==================== GOOGLE SHEETS CONFIG ====================
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1BE2le2ZVm2ej20w7UF5T7RSjO-V_Ii0RuhZQ2vEQQLY/edit"
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

def salvar_grafo_historico(id_usuario, query, articles_count, G, freq):
    """
    Salva topologia do grafo em nova aba do Google Sheets.
    Armazena: arestas (source, target, weight, salton) + metadados.
    """
    try:
        sheet = conectar_google_sheets()
        if not sheet:
            return False
        
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        safe_id = id_usuario[-8:] if len(id_usuario) > 8 else id_usuario
        tab_title = f"G_{safe_id}_{timestamp}"
        
        # Preparar dados das arestas com Salton
        edges_data = []
        for u, v in G.edges():
            weight = G[u][v].get('weight', 1)
            
            # Cálculo do Cosseno de Salton
            # Fórmula: Weight / sqrt(Freq(A) * Freq(B))
            freq_u = freq.get(u, 1)
            freq_v = freq.get(v, 1)
            
            if freq_u > 0 and freq_v > 0:
                salton = weight / np.sqrt(freq_u * freq_v)
            else:
                salton = 0
            
            # Adiciona com 4 casas decimais
            edges_data.append([u, v, weight, round(salton, 4)])
        
        metadata = [
            ["id_usuario", id_usuario],
            ["timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["query", query[:100]],
            ["articles_count", articles_count],
            ["nodes_count", len(G.nodes())],
            ["edges_count", len(G.edges())],
            ["---EDGES---", "", "", ""], # Ajuste para 4 colunas
            ["source", "target", "weight", "salton"] # Cabeçalho atualizado
        ]
        
        all_data = metadata + edges_data
        
        # Cria aba com 4 colunas (cols=4)
        worksheet = sheet.add_worksheet(
            title=tab_title, 
            rows=len(all_data) + 10, 
            cols=4 
        )
        worksheet.update(all_data)
        
        print(f"✅ Grafo salvo no Sheets: {tab_title} ({len(edges_data)} arestas)")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao salvar histórico: {e}")
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

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Delinéia",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS CUSTOMIZADO (BOTÕES VERDES) ====================
st.markdown("""
<style>
    /* Centralizar texto de expanders */
    .streamlit-expanderHeader {
        justify-content: center;
        text-align: center;
        font-weight: bold;
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

# ==================== FUNÇÕES DE INTERAÇÃO (MOVIDAS DO FINAL) ====================

def extract_concept_metadata(articles: list) -> dict:
    """
    Extrai metadados agregados (frequência, score médio, level médio) de cada conceito.
    """
    from collections import defaultdict
    
    concept_data = defaultdict(lambda: {'scores': [], 'levels': [], 'count': 0})
    
    for article in articles:
        for concept in article.get('concepts', []):
            name = concept.get('display_name', '')
            if name:
                concept_data[name]['scores'].append(concept.get('score', 0))
                concept_data[name]['levels'].append(concept.get('level', 0))
                concept_data[name]['count'] += 1
    
    metadata = {}
    for name, data in concept_data.items():
        metadata[name] = {
            'freq': data['count'],
            'score': sum(data['scores']) / len(data['scores']) if data['scores'] else 0,
            'level': sum(data['levels']) / len(data['levels']) if data['levels'] else 0
        }
    
    return metadata

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
        
        components.html(html_content, height=int(height.replace('px', '')) + 50, scrolling=False)
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
                help="Permite arrastar nós. Desative para layouts fixos."
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
                graphml_buffer = io.BytesIO()
                nx.write_graphml(G_filtered, graphml_buffer)
                graphml_buffer.seek(0)
                
                st.download_button(
                    "📥 GraphML (Gephi)",
                    data=graphml_buffer.getvalue(),
                    file_name="grafo_interativo.graphml",
                    mime="application/xml",
                    use_container_width=True,
                    help="Para Gephi ou Cytoscape"
                )
            except Exception as e:
                st.error(f"Erro: {e}")
        
        with col_exp2:
            edges_data = ["source,target,weight"]
            for u, v in G_filtered.edges():
                weight = G_filtered[u][v].get('weight', 1)
                edges_data.append(f"{u},{v},{weight}")
            
            csv_content = "\n".join(edges_data)
            
            st.download_button(
                "📥 Arestas (CSV)",
                data=csv_content,
                file_name="grafo_arestas.csv",
                mime="text/csv",
                use_container_width=True,
                help="Lista de conexões"
            )
        
        with col_exp3:
            nodes_data = ["node,degree,degree_centrality,selected"]
            degree_cent = nx.degree_centrality(G_filtered) if len(G_filtered.nodes()) > 0 else {}
            
            for node in G_filtered.nodes():
                deg = G_filtered.degree(node)
                dc = degree_cent.get(node, 0)
                sel = "sim" if node in selected_concepts else "não"
                nodes_data.append(f"{node},{deg},{dc:.4f},{sel}")
            
            csv_nodes = "\n".join(nodes_data)
            
            st.download_button(
                "📥 Nós (CSV)",
                data=csv_nodes,
                file_name="grafo_nos.csv",
                mime="text/csv",
                use_container_width=True,
                help="Lista de conceitos com métricas"
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
            label_visibility="collapsed"
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
                    help="Adiciona * ao final para recuperar variações"
                )
            
            with col_aspas:
                use_quotes = st.checkbox(
                    'Usar aspas (" ")',
                    value=True,
                    help="Coloca o termo entre aspas para busca exata"
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
            
            if st.button("➕ Colecionar termo", use_container_width=True, type="primary"):
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
                if st.button("AND", use_container_width=True, help="Interseção: retorna resultados que contenham TODOS os termos"):
                    st.session_state.search_key_text += " AND "
                    st.rerun()
            
            with col_or:
                if st.button("OR", use_container_width=True, help="União: retorna resultados que contenham QUALQUER um dos termos"):
                    st.session_state.search_key_text += " OR "
                    st.rerun()
            
            with col_not:
                if st.button("NOT", use_container_width=True, help="Exclusão: remove resultados que contenham o termo seguinte"):
                    st.session_state.search_key_text += " NOT "
                    st.rerun()
            
            with col_abre:
                if st.button("(", use_container_width=True, help="Abre parênteses para agrupar termos"):
                    st.session_state.search_key_text += "("
                    st.rerun()
            
            with col_fecha:
                if st.button(")", use_container_width=True, help="Fecha parênteses"):
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
                                if st.button(display_label, key=f"term_btn_{idx}", use_container_width=True):
                                    st.session_state.search_key_text += term
                                    st.rerun()
            
            col_limpar, col_limpar_termos = st.columns(2)
            with col_limpar:
                if st.button("🗑️ Limpar chave", use_container_width=True):
                    st.session_state.search_key_text = ""
                    st.rerun()
            with col_limpar_termos:
                if st.button("🗑️ Limpar termos coletados", use_container_width=True):
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
            
            if edited_key.strip():
                import json
                safe_text = json.dumps(edited_key.strip())
                
                copy_js = f"""
                <script>
                function copyToClipboard() {{
                    navigator.clipboard.writeText({safe_text}).then(function() {{
                        document.getElementById('copy-status').innerHTML = '✅ Copiado!';
                        setTimeout(function() {{
                            document.getElementById('copy-status').innerHTML = '';
                        }}, 2000);
                    }});
                }}
                </script>
                <div style="text-align: center;">
                    <button onclick="copyToClipboard()" style="
                        background-color: #ffffff;
                        color: #000000;
                        border: 1px solid #cccccc;
                        padding: 8px 16px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 14px;
                    ">📋 Copiar</button>
                    <span id="copy-status" style="margin-left: 10px; color: #21c354;"></span>
                </div>
                """
                components.html(copy_js, height=50)
            
            # Métricas
            col_info1, col_info2 = st.columns(2)
            col_info1.metric("Termos coletados", len(st.session_state.collected_terms))
            col_info2.metric("Caracteres", len(edited_key.strip()))
            
            st.divider()
            
            if st.button("📋 Copiar para o Painel", use_container_width=True, type="primary"):
                st.session_state.dashboard_query = edited_key.strip()
                st.session_state.dashboard_query_source = "construtor"
                st.success("✅ Chave copiada!")     
             
    rodape_institucional()

# ==================== ABAS PRINCIPAIS ====================
tab1, tab2, tab3, tab4 = st.tabs(["📚 Delineascópio", "🔬 Interação", "📜 Histórico", "📊 Painel"])

# ==================== ABA 1: DELINEASCÓPIO ====================
with tab1:
    st.title("📚 Delinéia - Delineamento de Escopo Temático")
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
                use_container_width=True
            )

            if submitted:
                if not all([nome, email, tema, questao, palavras_chave]):
                    st.error("⚠️ Por favor, preencha todos os campos obrigatórios (*)")
                else:
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
                            # Inicializar pipeline
                            pipe = ResearchScopePipeline(OPENALEX_EMAIL)

                            # Processar palavras-chave
                            kws = [k.strip() for k in palavras_chave.split(',') if k.strip()]

                            # Executar pipeline
                            tempo_inicio = time_module.time()
                            # Usa a função cacheada
                            st.session_state.resultado = run_cached_pipeline(nome, tema, questao, kws, genero)
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
                                    resultado = st.session_state.resultado
                                    salvar_grafo_historico(
                                        id_usuario=st.session_state.id_usuario,
                                        query=resultado.get('search_string', ''),
                                        articles_count=resultado.get('articles_count', 0),
                                        G=resultado.get('graph'),
                                        freq=resultado.get('concept_freq', {})
                                    )
                                except Exception as e:
                                    print(f"⚠️ Erro ao salvar histórico: {e}")

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
            st.header("🕸️ 2. Grafo de conceitos")
            st.caption("Etapa 2: Explore o grafo e o glossário antes de selecionar os conceitos")

            # Botão voltar
            if st.button("⬅️ Voltar ao Formulário"):
                st.session_state.step = 1
                st.rerun()

            # Informações do projeto (resumido)
            with st.expander("📋 Dados do Projeto", expanded=False):
                st.write(f"**Tema:** {d['tema']}")
                st.write(f"**Questão:** {d['questao']}")
                st.write(f"**Palavras-chave:** {d['palavras_chave']}")

            # Métricas
            col1, col2, col3 = st.columns(3)
            col1.metric("📚 Artigos Analisados", r.get('articles_count', 0))
            col2.metric("🧩 Conceitos no Grafo", r['graph_stats']['nodes'])
            col3.metric("🔗 Conexões", r['graph_stats']['edges'])

            # Layout: Grafo e Glossário lado a lado
            col_grafo, col_glossario = st.columns([1, 1])

            with col_grafo:
                st.subheader("🕸️ Grafo de Coocorrências")
                if r.get('visualization_path'):
                    st.image(r['visualization_path'], use_container_width=True)
                else:
                    st.warning("⚠️ Visualização não disponível")

            with col_glossario:
                st.subheader("📖 Glossário de Conceitos")
                with st.container(height=400):
                    st.markdown(r.get('glossary', '⚠️ Glossário não disponível'))

            # Instrução para próxima etapa
            st.divider()
            st.info("""
            💡 **Próximo passo:** Observe atentamente o grafo e o glossário acima. 
            Na próxima etapa, você selecionará os conceitos mais relevantes para sua pesquisa.
            Essa seleção será usada para gerar uma interpretação personalizada do grafo.
            """)

            # Botão avançar
            if st.button("Continuar para Seleção de Conceitos ▶️", type="primary", use_container_width=True):
                st.session_state.sub_step = 'b'
                st.rerun()

            rodape_institucional()

        # ========== SUB-ETAPA 2b: SELEÇÃO DE CONCEITOS ==========
        elif sub_step == 'b':
            top_concepts = r.get('top_concepts', [])[:9]

            st.header("🎯 3. Seleção de Conceitos")
            st.caption("Etapa 3: Escolha os conceitos mais relevantes para sua pesquisa")

            # Navegação
            if st.button("⬅️ Voltar ao Grafo"):
                st.session_state.sub_step = 'a'
                st.rerun()

            # Contexto
            primeiro_nome = d['nome'].split()[0]
            st.markdown(f"""
            ### {primeiro_nome}, quais conceitos do grafo são mais relevantes para seu projeto?

            Considerando seu tema **"{d['tema']}"**, selecione os conceitos que você considera 
            mais importantes para o delineamento do escopo da sua pesquisa.

            *Selecione pelo menos 1 conceito para continuar.*
            """)

            # Mostrar grafo como referência (menor)
            with st.expander("🕸️ Ver grafo novamente", expanded=False):
                if r.get('visualization_path'):
                    st.image(r['visualization_path'], use_container_width=True)

            # Seleção de conceitos com checkboxes
            st.subheader("📋 Conceitos Identificados na Rede")

            # Criar 3 colunas para os checkboxes
            cols = st.columns(3)
            selected = []

            for i, concept in enumerate(top_concepts):
                col_idx = i % 3
                with cols[col_idx]:
                    # Verificar se já estava selecionado antes
                    default_value = concept in st.session_state.get('selected_concepts', [])
                    if st.checkbox(concept, value=default_value, key=f"concept_{i}"):
                        selected.append(concept)

            # Atualizar session_state
            st.session_state.selected_concepts = selected

            # Contador
            st.divider()
            num_selected = len(selected)

            if num_selected == 0:
                st.warning("⚠️ Selecione pelo menos 1 conceito para continuar")
            else:
                st.success(f"✅ **{num_selected} conceito(s) selecionado(s):** {', '.join(selected)}")

            # Botão avançar (só habilitado se tiver seleção)
            
            col1, col2 = st.columns(2)

            with col2:
                if num_selected >= 1:
                    if st.button("Gerar Relatório de Delineamento ▶️", type="primary", use_container_width=True):
                        with st.spinner("🔄 Gerando relatório... (aguarde 1-2 minutos)"):
                            # Gerar conteúdo personalizado
                            from research_pipeline import GeminiQueryGenerator
                            gemini = GeminiQueryGenerator()

                            primeiro_nome = d['nome'].split()[0]
                            tema = d['tema']
                            original_kws = [k.strip() for k in d.get('palavras_chave', '').split(',') if k.strip()]
                            all_concepts = r.get('top_concepts', [])[:9]

                            # Gerar interpretação contextualizada
                            st.session_state.personalized_interpretation = gemini.generate_contextualized_interpretation(
                                tema, primeiro_nome, selected, all_concepts, genero=d.get('genero', 'Neutro')
                            )

                            # Gerar sugestões de palavras-chave
                            st.session_state.suggested_keywords = gemini.generate_keyword_suggestions(
                                tema, primeiro_nome, selected, original_kws
                            )

                            # Gerar chaves de busca (agora passando os termos ricos!)
                            st.session_state.suggested_strings = gemini.generate_search_strings(
                                tema, 
                                selected, 
                                original_kws,
                                st.session_state.suggested_keywords  # <-- NOVO PARÂMETRO
                            )

                            st.session_state.interpretation_generated = True
                            # Atualizar termos sugeridos no Sheets
                            if 'id_usuario' in st.session_state:
                                atualizar_termos_sugeridos(
                                    st.session_state.id_usuario,
                                    st.session_state.suggested_keywords
                                )

                        st.session_state.sub_step = 'c'
                        st.rerun()
                else:
                    st.button("Gerar Interpretação Personalizada ▶️", disabled=True, use_container_width=True)

            rodape_institucional()

        # ========== SUB-ETAPA 2c: INTERPRETAÇÃO PERSONALIZADA ==========
        elif sub_step == 'c':
            selected = st.session_state.get('selected_concepts', [])

            st.header("📋 4. Relatório")
            st.caption("Etapa 4: Interpretação baseada nos conceitos que você selecionou")

            # Navegação
            col_nav1, col_nav2 = st.columns([1, 3])
            with col_nav1:
                if st.button("⬅️ Voltar à Seleção"):
                    st.session_state.sub_step = 'b'
                    st.rerun()

            # Resumo da seleção
            st.success(f"✅ **Conceitos selecionados:** {', '.join(selected)}")

            # Informações do projeto
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

            # ========== SEÇÃO 1: AVALIAÇÃO INICIAL DO PROJETO ==========
            st.subheader("📋 Avaliação do Projeto")
            with st.container(border=True):
                st.markdown(r.get('full_report', '⚠️ Avaliação não disponível'))

            # ========== SEÇÃO 2: INTERPRETAÇÃO PERSONALIZADA ==========
            st.subheader("💡 Interpretação Personalizada do Grafo")
            with st.container(border=True):
                interpretation = st.session_state.get('personalized_interpretation', '')
                if interpretation:
                    st.markdown(interpretation)
                else:
                    st.markdown(r.get('graph_interpretation', '⚠️ Interpretação não disponível'))

            # ========== SEÇÃO 3: GRAFO ==========
            st.subheader("🕸️ Grafo de Coocorrências")
            if r.get('visualization_path'):
                st.image(r['visualization_path'], use_container_width=True)

            # ========== SEÇÃO 4: GLOSSÁRIO ==========
            st.subheader("📖 Glossário de Conceitos")
            with st.expander("Ver glossário completo", expanded=False):
                st.markdown(r.get('glossary', '⚠️ Glossário não disponível'))

            # ========== SEÇÃO 5: SUGESTÕES DE PALAVRAS-CHAVE ==========
            st.subheader("🔑 Sugestões de Palavras-chave")

            suggested_kws = st.session_state.get('suggested_keywords', [])

            if suggested_kws:
                for kw in suggested_kws:
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**{kw.get('term_en', 'N/A')}**")
                            st.caption(f"({kw.get('term_pt', 'N/A')})")
                        with col2:
                            st.write(kw.get('description', ''))
            else:
                st.info("Sugestões de palavras-chave não disponíveis")

            # ========== SEÇÃO 6: CHAVES DE BUSCA SUGERIDAS ==========
            st.subheader("🔎 Chaves de Busca Sugeridas")
            st.caption("Copie as chaves de busca abaixo para usar no Painel ou em bases de dados")

            suggested_strings = st.session_state.get('suggested_strings', {})

            if suggested_strings:
                for key, data in suggested_strings.items():
                    with st.container(border=True):
                        st.markdown(f"**{data.get('titulo', key)}**")
                        st.caption(data.get('descricao', ''))

                        col_str, col_btn = st.columns([4, 1])

                        with col_str:
                            st.code(data.get('string', ''), language='text')

                        with col_btn:
                            if st.button("📋 Copiar", key=f"copy_{key}", use_container_width=True):
                                st.session_state.dashboard_query = data.get('string', '')
                                st.session_state.dashboard_query_source = "delineascópio"
                                st.toast(f"✅ Chave de busca copiada para o Painel!")
            else:
                # Fallback: mostrar chave de busca original
                search_string = r.get('search_string', 'N/A')
                with st.container(border=True):
                    st.markdown("**🔎 Chave de Busca Original**")
                    col_str, col_btn = st.columns([4, 1])
                    with col_str:
                        st.code(search_string, language='text')
                    with col_btn:
                        if st.button("📋 Copiar", key="copy_original", use_container_width=True):
                            st.session_state.dashboard_query = search_string
                            st.session_state.dashboard_query_source = "delineascópio"
                            st.toast("✅ Chave de busca copiada para o Painel!")

            # ========== SEÇÃO 7: CHAVE DE TRANSPARÊNCIA (ORIGINAL OPENALEX) ==========
            st.subheader("🔬 Transparência: Chave de Busca Usada")
            st.caption("Esta é a chave de busca exata que foi usada para recuperar artigos do OpenAlex")
            
            with st.container(border=True):
                # Mostrar objetivo da busca
                search_objective = r.get('search_objective', '')
                if search_objective:
                    st.markdown(f"**Objetivo:** {search_objective}")
                    st.divider()
                
                # Mostrar chave original
                search_string = r.get('search_string', 'N/A')
                st.markdown("**Chave de busca executada:**")
                
                col_str, col_btn = st.columns([4, 1])
                
                with col_str:
                    st.code(search_string, language='text')
                
                with col_btn:
                    if st.button("📋 Copiar", key="copy_transparency", use_container_width=True):
                        st.session_state.dashboard_query = search_string
                        st.session_state.dashboard_query_source = "delineascópio"
                        st.toast("✅ Chave de busca copiada para o Painel!")
                
                # Estatísticas
                articles_count = r.get('articles_count', 0)
                graph_stats = r.get('graph_stats', {})
                
                st.caption(f"📊 Resultados: {articles_count} artigos encontrados | "
                          f"{graph_stats.get('nodes', 0)} conceitos | "
                          f"{graph_stats.get('edges', 0)} coocorrências")
            
            # ========== SEÇÃO 8: AÇÕES FINAIS ==========
            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                # PDF disponível após completar a trilha
                try:
                    # Gerar PDF com todos os dados da trilha
                    pdf_bytes = generate_pdf_report(
                        form_data=d,
                        result=r,
                        selected_concepts=selected,
                        suggested_keywords=st.session_state.get('suggested_keywords', []),
                        suggested_strings=st.session_state.get('suggested_strings', {}),
                        badges=st.session_state.get('badges', [])
                    )
                    st.download_button(
                        "📥 Baixar PDF Completo",
                        pdf_bytes,
                        f"delineamento_{d['nome'].replace(' ', '_')}.pdf",
                        "application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {str(e)}")

            with col2:
                if st.button("📊 Ir ao Painel", use_container_width=True):
                    st.info("💡 Use as chaves de busca sugeridas para explorar mais a literatura no Painel!")

            with col3:
                if st.button("📝 Avaliar Sistema", type="primary", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()

            # Dica final
            st.info("""
            🎉 **Parabéns!** Você completou a trilha de delineamento!

            Agora você pode:
            - 📥 **Baixar o PDF** com o relatório completo
            - 📊 **Usar o Painel** para explorar mais a literatura
            - 📝 **Avaliar o sistema** e nos ajudar a melhorar
            """)

            # Botão novo projeto
            if st.button("🔄 Iniciar Novo Delineamento", use_container_width=True):
                st.session_state.step = 1
                st.session_state.resultado = None
                st.session_state.form_data = {}
                st.session_state.avaliacao_completa = False
                st.session_state.badges = []
                st.session_state.sub_step = 'a'
                st.session_state.selected_concepts = []
                st.session_state.interpretation_generated = False
                st.session_state.personalized_interpretation = None
                st.session_state.suggested_keywords = []
                st.session_state.suggested_strings = {}
                st.rerun()

                limpar_memoria()

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
                help="Clique para baixar o Termo de Consentimento Livre e Esclarecido completo"
            )

        st.markdown("") # Um pequeno espaço
        tcle_aceite = st.checkbox(
            "📝 **Li, compreendi e CONCORDO em participar da Etapa 1 (formulários online).**",
            key="tcle_aceite"
        )

        st.markdown("") # Um pequeno espaço
        tcle_rejeita = st.checkbox(
            "📝 **Li, mas NÃO CONCORDO em participar desta pesquisa.**",
            key="tcle_rejeita"
        )

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
                value=5,
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
                "✅ **CONCORDO em ser convidado(a) para atividades com gravação de áudio e vídeo.**",
                key="aceite_continuidade",
                help="Ao marcar esta opção, você demonstra interesse em contribuir com o desenvolvimento do Delinéia"
            )

            rejeita_continuidade = st.checkbox(
                "✅ **NÃO CONCORDO em participar de atividades qualitativas com gravação.**",
                key="rejeita_continuidade",
                help="Você não será considerado em convites de continuidade da pesquisa."
            )

            if aceite_continuidade:
                st.success("🎉 Obrigado por aceitar continuar conosco! Você receberá um e-mail com mais informações em breve.")

            st.divider()

            # ==================== BOTÃO DE ENVIO ====================
            submitted = st.form_submit_button(
                "📤 Enviar Avaliação",
                type="primary",
                use_container_width=True
            )

            if submitted:
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
                st.success("✅ Avaliação enviada com sucesso!")
                st.balloons()

                # Resumo da avaliação
                continuidade_msg = "Sim ✅" if aceite_continuidade else "Não"
                
                st.info(f"""
                📊 **Resumo da sua avaliação:**

                • **NPS:** {nps}/10 ({nps_category})
                • **Nível acadêmico:** {q27}
                • **Experiência bibliométrica:** {q28}
                • **Área:** {q29}
                • **Tempo de uso:** {q30}
                • **Aceite para continuidade:** {continuidade_msg}

                🏆 **Badge desbloqueado:** Avaliador

                Obrigado por dedicar seu tempo para avaliar o Delinéia!
                Seu feedback é essencial para o desenvolvimento contínuo do sistema.
                """)

                # Avançar para próxima etapa
                st.session_state.step = 4
                st.rerun()

        rodape_institucional()
    
    # ========== ETAPA 4: CONCLUSÃO ==========
    elif st.session_state.step == 4:
        st.success("🎉 Parabéns! Você completou todas as etapas!")
        st.markdown(f"### 🏆 Conquista Desbloqueada: {g('Delineador', 'Delineadora')}!")
        st.balloons()

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
        with st.expander("📜 Créditos e Informações"):
            st.markdown("""
            <div style="text-align: center; 
                        background-color: #f8f9fa; 
                        padding: 1.5rem; 
                        border-radius: 0.5rem;
                        color: #000000;">
            
            **Título:** A palavra no escuro ou os dialetos do poço
                        
            **Álbum:** Os olhos de Borges (Versão musical do livro homônimo)
                        
            **Livro:** BRASIL, J.V. *Os olhos de Borges*. Porto Alegre: WS Editor, 1997.
                        
            **Autoria:** Jaime Vaz Brasil
                        
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
            if st.button("📜 Leia o prólogo da tese", use_container_width=True):
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

Um ensaio sobre Erico Verissimo e Graham Greene menciona o "[...] delineamento de dois perfis de personagens [...]".[^10] Um estudo sobre Machado de Assis foca no "[...] delineamento do percurso da escrita de 'O alienista' [...]",[^11] analisando como Machado de Assis esboçou e refez sua obra. Um crítico, sobre o poeta Demétrio Vieira Diniz, afirma que seu livro "[...] atesta e faz saber o delineamento de uma singular dicção".[^12] É a palavra que usamos para entender a criação, perfeita para descrever o processo de descoberta que o aluno, no centro desta tese, está colocado a realizar.

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

        if st.button("🔄 Iniciar Novo Delineamento", use_container_width=True):
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
        # ========================================================
        # 🔒 LÓGICA DE PRIVACIDADE E FILTRO DE USUÁRIO
        # ========================================================
        grafos_salvos = []
        user_id_atual = st.session_state.get('id_usuario')

        if user_id_atual:
            # Se temos usuário logado/identificado, filtramos pelo ID dele
            grafos_salvos = exp.listar_grafos_salvos(sheet, user_id_atual)
                    
            if not grafos_salvos:
                st.info(f"Nenhum histórico encontrado para seu usuário atual. Salve um grafo na aba 'Exportação' primeiro.")
        else:
            # Se não tem usuário identificado, não mostra nada (Privacidade)
            st.warning("⚠️ Você precisa preencher o Formulário Inicial (Etapa 1) para acessar seu histórico privado.")
            st.stop() # Para a execução desta aba aqui para proteger dados
                
        # Se passou daqui, é porque tem grafos e é o usuário certo

        if grafos_salvos:
            st.subheader("1. Selecione os Delineamentos para Comparar")
                    
            # Layout de seleção
            col_sel1, col_sel2 = st.columns(2)
            opcoes = [g['title'] for g in grafos_salvos]
                    
            with col_sel1:
                # Padrão: Penúltimo grafo (se existir) ou o mesmo
                idx_a = len(opcoes)-1 if len(opcoes) > 1 else 0
                g1_title = st.selectbox("Delineamento A (Referência/Antigo):", options=opcoes, index=idx_a)
                    
            with col_sel2:
                # Padrão: Grafo mais recente (primeiro da lista)
                g2_title = st.selectbox("Delineamento B (Atual/Recente):", options=opcoes, index=0)

            # Botão de Ação
            if st.button("🔄 Comparar Delineamentos", type="primary", use_container_width=True):
                if g1_title == g2_title:
                    st.warning("⚠️ Selecione dois delineamentos diferentes para ver as diferenças.")
                else:
                    with st.spinner("⏳ Baixando dados do Google Sheets e calculando similaridade..."):
                        # Resgata os objetos worksheet correspondentes
                        ws1 = next(g['obj'] for g in grafos_salvos if g['title'] == g1_title)
                        ws2 = next(g['obj'] for g in grafos_salvos if g['title'] == g2_title)
                                
                        # Carrega os dados usando função do export_utils
                        df1 = exp.carregar_grafo_do_sheets(ws1)
                        df2 = exp.carregar_grafo_do_sheets(ws2)
                                
                        if df1 is not None and df2 is not None:
                            # Calcula a mágica
                            metrics = exp.calcular_comparacao(df1, df2)
                                    
                            # --- RESULTADOS ---
                            st.divider()
                            st.subheader("📊 Resultados da Comparação")
                                    
                            # 1. Métricas Principais
                            col_res1, col_res2, col_res3 = st.columns(3)
                                    
                            col_res1.metric(
                                "Similaridade (Jaccard)", 
                                f"{metrics['jaccard']*100:.1f}%",
                                help="Mede o quanto os vocabulários se sobrepõem (0% = totalmente diferentes, 100% = iguais)."
                            )
                                    
                            delta = metrics['qtd_2'] - metrics['qtd_1']
                            col_res2.metric(
                                "Tamanho do Vocabulário", 
                                f"{metrics['qtd_2']} conceitos",
                                f"{delta:+}",
                                help="Diferença no número total de conceitos entre B e A."
                            )
                                    
                            col_res3.metric(
                                "Novos Conceitos", 
                                len(metrics['exclusivos_novos']),
                                help="Conceitos que existem em B mas não existiam em A."
                            )
                                    
                            # 2. Detalhamento Semântico
                            st.markdown("---")
                            c1, c2 = st.columns(2)
                                    
                            with c1:
                                st.markdown("#### 🆕 O que entrou (Novidades)")
                                if metrics['exclusivos_novos']:
                                    # Mostra tags coloridas ou lista
                                    st.success(", ".join(metrics['exclusivos_novos'][:50]))
                                    if len(metrics['exclusivos_novos']) > 50:
                                        st.caption(f"...e mais {len(metrics['exclusivos_novos'])-50} conceitos.")
                                else:
                                    st.info("Nenhum conceito novo adicionado.")

                            with c2:
                                st.markdown("#### 🗑️ O que saiu (Removidos)")
                                if metrics['exclusivos_antigos']:
                                    st.error(", ".join(metrics['exclusivos_antigos'][:50]))
                                    if len(metrics['exclusivos_antigos']) > 50:
                                        st.caption(f"...e mais {len(metrics['exclusivos_antigos'])-50} conceitos.")
                                else:
                                    st.info("Nenhum conceito foi removido.")
                                            
                            # 3. Intersecção (O que permaneceu)
                            with st.expander(f"🤝 Núcleo Estável ({len(metrics['comuns'])} conceitos mantidos)"):
                                st.write(", ".join(metrics['comuns']))

                            # ================== COMPARAÇÃO 🍒 (Versão com Rerun Forçado) ==================
                            st.divider()
                            st.markdown("### 🤖 O que o Delinéia diz sobre sua evolução?")
                                    
                            # Só mostra se houver diferença e se a análise AINDA NÃO foi feita
                            if metrics['jaccard'] < 0.99:
                                        
                                # 1. MOSTRAR BOTÃO (Se não tiver análise salva)
                                if 'ultima_analise_historico' not in st.session_state:
                                    if st.button("✨ Gerar Análise Pedagógica da Mudança", type="primary", use_container_width=True, key="btn_analise_ia_tab8"):
                                                
                                        nome_aluno = st.session_state.form_data.get('nome', 'Pesquisador').split()[0]
                                        genero_aluno = st.session_state.form_data.get('genero', 'Neutro')
                                                
                                        with st.spinner(f"🧠 O Orientador IA está analisando a trajetória de {nome_aluno}..."):
                                            try:
                                                if 'gemini_gen' not in st.session_state:
                                                    from research_pipeline import GeminiQueryGenerator
                                                    st.session_state.gemini_gen = GeminiQueryGenerator()
                                                        
                                                # Gera
                                                analise = st.session_state.gemini_gen.generate_evolution_analysis(
                                                    metrics, 
                                                    nome_aluno, 
                                                    genero=genero_aluno 
                                                )
                                                        
                                                # Salva
                                                st.session_state['ultima_analise_historico'] = analise
                                                        
                                                # FORÇA O RECARREGAMENTO PARA EXIBIR IMEDIATAMENTE
                                                st.rerun()
                                                        
                                            except Exception as e:
                                                st.error(f"Erro na conexão com IA: {str(e)}")

                                # 2. MOSTRAR RESULTADO (Se já tiver análise salva)
                                else:
                                    st.markdown("### 📝 Parecer da Orientação Artificial")
                                    st.info(st.session_state['ultima_analise_historico'], icon="🤖")
                                            
                                    # Botão para limpar e fazer de novo
                                    if st.button("🔄 Refazer Análise / Limpar", key="btn_limpar_analise"):
                                        del st.session_state['ultima_analise_historico']
                                        st.rerun()
                            else:
                                st.info("Os dois delineamentos são idênticos. Mude a busca e gere um novo grafo para ver a evolução.")    
                        else:
                            st.error("Erro ao ler os dados das planilhas. Verifique se as abas contêm dados válidos.")
    else:
        st.error("Não foi possível conectar ao Google Sheets.")

# ==================== ABA 4: PAINEL DE ANÁLISE ====================
with tab4:
    st.title("📊 Painel de Exploração de Dados")
    st.caption("Análise profunda dos dados do OpenAlex")

    # Sidebar para configuração
    with st.sidebar:
        st.header("⚙️ Configurar Busca")

        # Campo de busca
        query = st.text_input(
            "Chave de Busca:",
            value=st.session_state.get('dashboard_query', "games AND education"),
            help="Use operadores: AND, OR, NOT"
        )

        if 'dashboard_query' in st.session_state and st.session_state.dashboard_query:
            source = st.session_state.get('dashboard_query_source', 'delineascópio')
            if source == "construtor":
                st.info("📋 Chave de busca copiada do construtor de chaves")
            else:
                st.info("📋 Chave de busca copiada do Delineascópio")

        st.divider()
        st.subheader("🔧 Filtros")

        # Opção de sincronizar configurações
        with st.expander("⚙️ Configurações Avançadas"):
            sync_config = st.checkbox("Usar configuração padrão", value=True)

            if sync_config:
                st.info("**Configuração Padrão:**\n- Limite: 500 artigos\n- Score mínimo: 0.35\n- Level mínimo: 0")
                limit = 500
                min_score = 0.35
                min_level = 0
            else:
                limit = st.slider("Limite de artigos:", 10, 500, 100, 10,
                    help="Número máximo de artigos a buscar na API OpenAlex")
                min_score = st.slider("Score mínimo:", 0.0, 1.0, 0.35, 0.05,
                    help="Relevância mínima do conceito (0-1). Valores maiores = conceitos mais relevantes")
                min_level = st.slider("Level mínimo:", 0, 5, 0, 1,
                    help="Nível hierárquico do conceito (0-5). 0 = geral, 5 = muito específico")

        min_cooc = st.slider("Coocorrência mínima:", 1, 10, 2, 1,
            help="Frequência mínima de coocorrência para formar aresta no grafo")

        st.divider()

        # Botão de buscar
        if st.button("🔍 Buscar", type="primary", use_container_width=True):
            limpar_memoria()
            with st.spinner("🔄 Em processamento... confira no Painel   "):
                try:
                    # 1. BUSCA (CACHEADA)
                    # Passamos 0,0 no score/level para o cache guardar TUDO.
                    # A filtragem fina acontece visualmente abaixo, sem bater na API de novo.
                    raw_articles = search_openalex_cached(query, limit, 0, 0)
                    
                    # 2. FILTRAGEM LOCAL (PYTHON)
                    # Aqui aplicamos os sliders de Score e Level nos dados que já estão na memória
                    filtered_concepts_lists = []
                    
                    for article in raw_articles:
                        # Extrai apenas conceitos que passam no filtro dos sliders
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

                    # 4. DATAFRAME LIMPO (Usando a nova função auxiliar)
                    df_display = process_openalex_dataframe(raw_articles)

                    # Salvar no Session State
                    st.session_state.dashboard_data = {
                        'articles': raw_articles,      # Guarda o bruto (JSON)
                        'df_display': df_display,      # Guarda a tabela limpa
                        'concepts_lists': filtered_concepts_lists,
                        'graph': G
                    }

                    # Resumo no Expander
                    with st.expander("📋 Detalhes da Busca", expanded=False):
                        st.write(f"**Chave de busca:** `{query}`")
                        st.write(f"**Limite:** {limit}")
                        st.write(f"**Coocorrência mínima:** {min_cooc}")
                        st.write(f"**Filtros:** Score ≥ {min_score} | Level ≥ {min_level}")
                        st.write(f"**Artigos recuperados:** {len(raw_articles)}")
                        st.write(f"**Conceitos extraídos:** {len(filtered_concepts_lists)} listas válidas")
                        st.write(f"**Nós no grafo:** {len(G.nodes())}")

                    st.success(f"✅ Análise concluída: {len(raw_articles)} artigos | {len(G.nodes())} nós no grafo")

                except Exception as e:
                    st.error(f"❌ Erro na busca: {str(e)}")
                    # Dica de debug útil em dev
                    # st.exception(e)
        
        st.divider()

        # ========== SEÇÃO SOBRE ==========
        with st.expander("📋 Sobre o Delinéia"):
            st.markdown("""
            ### O que é o Delinéia?
            O Delinéia é um sistema de apoio ao delineamento do escopo temático de projetos de pesquisa no ensino superior e foi desenvolvido como parte de uma tese de doutorado em Informática na Educação. O sistema combina inteligência artificial generativa (Google Gemini) com análise bibliométrica de coocorrência de palavras (OpenAlex) para auxiliar estudantes de graduação e de pós-graduação no esboço de seus projetos de pesquisa.
        
            ### Desenvolvimento
            **Autor:** Rafael Antunes dos Santos  
            
            **Instituição:**             
            - Universidade Federal do Rio Grande do Sul (UFRGS) 
            - Centro Interdisciplinar de Novas Tecnologias na Educação (Cinted)
            - Programa de Pós-Graduação em Informática na Educação (PPGIE)
              
            **Nível:** Doutorado  
            **Orientador:** Prof. Dr. Eliseo Berni Reategui  
        
            **Formação Anterior:**
            - Mestre em Comunicação e Informação pela UFRGS (PPGCOM)  
            - Bacharel em Biblioteconomia pela UFRGS (DCI/FABICO) - CRB10/1898
        
            **Currículo Lattes:** [http://lattes.cnpq.br/5228660998907867](http://lattes.cnpq.br/5228660998907867)
        
            ### Abordagem Interdisciplinar
            Este projeto situa-se no diálogo entre Informática na Educação e Ciência da Informação, explorando como tecnologias de IA podem apoiar processos de pesquisa científica no ensino superior.
        
            ### Funcionalidades
            - **Delineascópio:** Feedback personalizado sobre projetos de pesquisa        
            - **Interação:** Grafo dinâmico
              - **Movimentação com física:** Inclusão e exclusão de nós
              - **Exportação de rede:** Dados em GraphML e CSV
              - **Construtor de chaves de busca:** Para maior autonomia
            - **Histórico:** Comparação entre grafos
              - **Análise pedagógica da mudança
            - **Painel:** Análise profunda de dados do OpenAlex:
              - **Artigos:** Contagens de artigos e links de acesso
              - **Conceitos:** Contagens de conceitos, nuvem de palavras e Lei de Zipf
              - **Coocorrências:** Contagens de associações entre conceitos e matrizes
              - **Grafo:** Visualização aumentada
              - **Mapa Temático:** Posição do cluster
              - **Estatísticas:** Resumo breve
              - **Exportação:** Dados em JSON, CSV, GraphML, .net, XLSX, BibTeX e RIS
            
            ### Tecnologias
            - Python | Streamlit
            - Google Gemini AI 2.5 Pro | Anthropic Claude Opus 4.5
            - OpenAlex API
            - NetworkX | Plotly | PyVis | ReportLab
        
            ### Contato
            📧 rafael.antunes@ufrgs.br
            📧 rderafa@gmail.com           
        
            ### Versão
            Delinéia I - 2025

            ### Agradecimentos
            Ao **Orientador** Eliseo Berni Reategui; Aos **Professores** Alexandra Lorandi, Alexandre Ribas Semeler, Dante Augusto Couto Barone, Elisa Boff, Fernando Becker, Gabriela Trindade Perry, Ida Regina Chitto Stumpf, Leandro Krug Wives, Marcus Vinicius de Azevedo Basso, Maria de Fátima Santos Maia, Milton Antonio Zaro, Patrícia Fernanda da Silva, Rafael Port da Rocha, Regina Helena Van der Laan, Renato Ventura Bayan Henriques, Rosa Maria Vicari, Samile Andréa de Souza Vanz, Sérgio Roberto Kieling Franco, Sonia Elisa Caregnato e Vanessa Soares Maurente. Aos colegas do grupo de pesquisa **GTech.Edu** e à **CAPES**, pela concessão de bolsa de estudos.
            """)
    
    # Área principal do painel
    if st.session_state.dashboard_data is None:
        st.info("👈 Configure os parâmetros na barra lateral e clique em **Buscar** para iniciar a análise")

        # Mostrar exemplo
        with st.expander("💡 Exemplo de uso"):
            st.markdown("""
            **Como usar o Painel:**

            1. **Digite uma chave de busca** (ex: "machine learning" AND education)
            2. **Ajuste os filtros** conforme necessário
            3. **Clique em Buscar** para processar
            4. **Explore as abas** com diferentes análises
            5. **Exporte os dados** quando necessário

            **Dica:** Você pode copiar as chaves de busca do Delineascópio!
            """)

    else:
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
            st.header("📚 Artigos Analisados")
            
            # Recupera o DataFrame limpo que geramos no botão Buscar
            # Isso evita ter que recriar o DataFrame a cada clique na aba
            df_display = st.session_state.dashboard_data.get('df_display')
            
            if df_display is not None and not df_display.empty:
                st.metric("Total de Artigos", len(df_display))
                
                # Exibe Tabela Interativa
                st.dataframe(
                    df_display,
                    use_container_width=True,
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

        # ========== SUB-ABA 2: CONCEITOS ==========
        with t2:
            st.header("🧩 Análise de Conceitos")

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
                
                st.plotly_chart(fig_cloud, use_container_width=True)
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

            # Gráfico de barras
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

            st.plotly_chart(fig, use_container_width=True)

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
                A **Lei de Zipf** prediz que a frequência de uma palavra é inversamente proporcional
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

                st.plotly_chart(fig_zipf, use_container_width=True)

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
                    """)

            # Tabela
            st.subheader("📋 Tabela de Frequências")
            st.dataframe(df_freq, width="stretch")

            st.divider()

            # Distribuição
            st.subheader("📊 Distribuição de Conceitos por Artigo")

            concepts_per_article = [len(c) for c in concepts_lists]

            fig2 = px.histogram(
                x=concepts_per_article,
                nbins=20,
                labels={'x': 'Número de conceitos', 'y': 'Frequência'},
                title="Distribuição de Conceitos por Artigo"
            )

            st.plotly_chart(fig2, use_container_width=True)

            if len(concepts_per_article) > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("Média", f"{sum(concepts_per_article)/len(concepts_per_article):.1f}")
                col2.metric("Mínimo", min(concepts_per_article))
                col3.metric("Máximo", max(concepts_per_article))

        # ========== SUB-ABA 3: COOCORRÊNCIAS ==========
        with t3:
            st.header("🔗 Análise de Coocorrências")

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

            st.plotly_chart(fig, use_container_width=True)

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
            
            st.plotly_chart(fig_salton, use_container_width=True)
            
            # Botão para baixar matriz completa
            with st.expander("💾 Baixar Matriz Completa de Salton"):
                st.caption("Matriz com todos os conceitos do grafo")
                
                all_concepts = list(freq.keys())
                full_salton = pd.DataFrame(0.0, index=all_concepts, columns=all_concepts)
                
                for (c1, c2), f in pairs.items():
                    salton_value = f / np.sqrt(concept_freq.get(c1, 1) * concept_freq.get(c2, 1))
                    full_salton.loc[c1, c2] = round(salton_value, 4)
                    full_salton.loc[c2, c1] = round(salton_value, 4)
                
                csv_salton = full_salton.to_csv()
                
                st.download_button(
                    "📥 Baixar Matriz Salton (CSV)",
                    data=csv_salton,
                    file_name="matriz_salton_completa.csv",
                    mime="text/csv",
                    use_container_width=True
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

            st.plotly_chart(fig3, use_container_width=True)

        # ========== SUB-ABA 4: GRAFO ==========
        with t4:
            st.header("🕸️ Análise do Grafo")

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

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # Comunidades
                st.subheader("👥 Detecção de Comunidades (Cluster)")

                try:
                    from networkx.algorithms import community
                    communities = list(community.greedy_modularity_communities(G))

                    st.metric("Número de Comunidades", len(communities))

                    for i, comm in enumerate(communities, 1):
                        with st.expander(f"Comunidade {i} ({len(comm)} conceitos)"):
                            members = list(comm)[:20]
                            st.write(", ".join(members))
                            if len(comm) > 20:
                                st.caption(f"... e mais {len(comm)-20} conceitos")

                except Exception as e:
                    st.info("Não foi possível detectar comunidades")

                st.divider()

                # Visualização interativa
                st.subheader("🎨 Visualização Interativa")

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

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("⚠️ Grafo muito grande (>100 nós). Use filtros para reduzir o tamanho.")

        # ========== SUB-ABA 5: MAPA TEMÁTICO =========
        with t5:
            st.header("🗺️ Mapa Temático (Diagrama Estratégico)")

        @st.cache_data
        def cached_thematic_map(graph_data, concepts_lists, method, min_size):

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

                                st.plotly_chart(fig_mapa, use_container_width=True)

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
                                            concepts_display = ", ".join(cluster["conceitos"][:10])
                                            if len(cluster["conceitos"]) > 10:
                                                concepts_display += f" ... (+{len(cluster['conceitos']) - 10} mais)"
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
            st.header("📊 Estatísticas Completas")

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

                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if len(G.edges()) > 0:
                    weights = [d['weight'] for u, v, d in G.edges(data=True)]

                    fig = px.histogram(
                        x=weights,
                        nbins=20,
                        labels={'x': 'Peso', 'y': 'Frequência'},
                        title="Distribuição dos Pesos das Arestas"
                    )

                    st.plotly_chart(fig, use_container_width=True)

        # ========== SUB-ABA 7: EXPORTAÇÃO (ATUALIZADA) ==========
        with t7:
            st.header("💾 Exportação de Dados")

            col1, col2, col3 = st.columns(3)

            # --- COLUNA 1: JSON ---
            with col1:
                st.subheader("📄 JSON")

                st.download_button(
                    "📥 Artigos (JSON Completo)",
                    json.dumps(articles, indent=2, ensure_ascii=False),
                    "articles.json",
                    "application/json",
                    help="Dados brutos completos (ideal para mineração).",
                    use_container_width=True
                )

                st.download_button(
                    "📥 Conceitos (JSON)",
                    json.dumps(concepts_lists, indent=2, ensure_ascii=False),
                    "concepts.json",
                    "application/json",
                    use_container_width=True
                )

                cooc_json = [
                    {"conceito1": c1, "conceito2": c2, "frequencia": f}
                    for (c1, c2), f in pairs.items()
                ]

                st.download_button(
                    "📥 Coocorrências (JSON)",
                    json.dumps(cooc_json, indent=2, ensure_ascii=False),
                    "cooccurrences.json",
                    "application/json",
                    use_container_width=True
                )

            # --- COLUNA 2: CSV ---
            with col2:
                st.subheader("📊 CSV")

                df_articles_export = pd.DataFrame([
                    {
                        'title': a.get('title', ''),
                        'year': a.get('year', ''),
                        'num_concepts': len(a.get('concepts', []))
                    }
                    for a in articles
                ])

                st.download_button(
                    "📥 Artigos (CSV)",
                    df_articles_export.to_csv(index=False),
                    "articles.csv",
                    "text/csv",
                    use_container_width=True
                )

                df_concepts = pd.DataFrame(
                    freq.most_common(),
                    columns=['concept', 'frequency']
                )

                st.download_button(
                    "📥 Conceitos (CSV)",
                    df_concepts.to_csv(index=False),
                    "concepts.csv",
                    "text/csv",
                    use_container_width=True
                )

                edges_list = [[u, v, d['weight']] for u, v, d in G.edges(data=True)]
                df_cooc = pd.DataFrame(edges_list, columns=['source', 'target', 'weight'])

                st.download_button(
                    "📥 Coocorrências (CSV)",
                    df_cooc.to_csv(index=False),
                    "cooccurrences.csv",
                    "text/csv",
                    use_container_width=True
                )

            # --- COLUNA 3: OUTROS FORMATOS (GraphML e .net) ---
            with col3:
                st.subheader("🔧 Redes")

                # 1. GraphML (Já existente)
                import tempfile
                try:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.graphml') as f:
                        nx.write_graphml(G, f.name)
                        # Reabre para leitura
                    with open(f.name, 'r', encoding='utf-8') as file:
                        graphml_content = file.read()
                    
                    st.download_button(
                        "📥 Grafo (GraphML)",
                        graphml_content,
                        "graph.graphml",
                        "application/xml",
                        help="Para Gephi ou Cytoscape",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro GraphML: {e}")

                # 2. Pajek .net (NOVO - INSERIDO AQUI)
                try:
                    # Usa a função nova que criamos no export_utils
                    pajek_data = exp.generate_pajek_net(G)
                    st.download_button(
                        "📥 Grafo (.net Pajek)", 
                        pajek_data, 
                        "graph.net", 
                        "text/plain",
                        help="Para VOSviewer ou Pajek",
                        use_container_width=True
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
                excel_data = exp.generate_excel(articles)
                with col_exp1:
                    st.download_button(
                        label="📊 Excel (.xlsx)",
                        data=excel_data,
                        file_name="delineia_resultados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Planilha formatada com conceitos, score e level.",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Erro Excel: {e}")
            
            # 2. Botão BibTeX
            try:
                bibtex_data = exp.generate_bibtex(articles)
                with col_exp2:
                    st.download_button(
                        label="🎓 BibTeX (.bib)",
                        data=bibtex_data,
                        file_name="delineia_referencias.bib",
                        mime="text/plain",
                        help="Para LaTeX/Overleaf.",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Erro BibTeX: {e}")
                
            # 3. Botão RIS
            try:
                ris_data = exp.generate_ris(articles)
                with col_exp3:
                    st.download_button(
                        label="📚 RIS (Zotero)",
                        data=ris_data,
                        file_name="delineia_referencias.ris",
                        mime="application/x-research-info-systems",
                        help="Para Zotero, Mendeley, EndNote.",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Erro RIS: {e}")

            # --- PACOTE ZIP ---
            st.subheader("📦 Pacote Completo")

            if st.button("🎁 Gerar ZIP com Todos os Dados", use_container_width=True):
                with st.spinner("📦 Gerando arquivo ZIP..."):
                    import zipfile
                    from io import BytesIO
                    from datetime import datetime
                    
                    zip_buffer = BytesIO()

                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        # JSON
                        zf.writestr('articles.json', json.dumps(articles, indent=2, ensure_ascii=False))
                        zf.writestr('concepts.json', json.dumps(concepts_lists, indent=2, ensure_ascii=False))
                        zf.writestr('cooccurrences.json', json.dumps(cooc_json, indent=2, ensure_ascii=False))

                        # CSV
                        zf.writestr('articles.csv', df_articles_export.to_csv(index=False))
                        zf.writestr('concepts.csv', df_concepts.to_csv(index=False))
                        zf.writestr('cooccurrences.csv', df_cooc.to_csv(index=False))

                        # Redes
                        zf.writestr('graph.graphml', graphml_content)
                        # Adicionando o .net ao ZIP também
                        try:
                            if isinstance(pajek_data, bytes):
                                zf.writestr('graph.net', pajek_data)
                            else:
                                zf.writestr('graph.net', pajek_data.encode('utf-8'))
                        except:
                            pass # Se falhar o pajek, gera o zip sem ele

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
                        use_container_width=True
                    )
        
    rodape_institucional()