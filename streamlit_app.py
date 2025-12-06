# -*- coding: utf-8 -*-

import streamlit as st
from datetime import datetime
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

# ==================== GOOGLE SHEETS CONFIG ====================
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1BE2le2ZVm2ej20w7UF5T7RSjO-V_Ii0RuhZQ2vEQQLY/edit"
ABA_FORMULARIO_INICIAL = "formulario_inicial"
ABA_RESULTADOS_PIPELINE = "resultados_pipeline"
ABA_FORMULARIO_AVALIACAO = "formulario_avaliacao"

@st.cache_resource(show_spinner=False)
def conectar_google_sheets():
    """
    Conecta ao Google Sheets usando credenciais do Streamlit Secrets
    CORREÇÃO APLICADA: Tratamento da private_key para converter 
    \\n literal em quebras de linha reais.
    """
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    try:
        # Ler credenciais DOS SECRETS
        google_creds = st.secrets["google_credentials"]
        
        # Converter para dict
        creds_dict = dict(google_creds)
        
        if "private_key" in creds_dict:
            # Primeiro tenta substituir \\n (escaped)
            pk = creds_dict["private_key"]
            if "\\n" in pk:
                pk = pk.replace("\\n", "\n")
            creds_dict["private_key"] = pk
        
        # Criar credenciais a partir do dict
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

def enviar_formulario_inicial(form_data):
    """Envia dados do formulário inicial para Google Sheets"""
    try:
        sheet = conectar_google_sheets()
        
        if not sheet:
            return None
        
        worksheet = sheet.worksheet(ABA_FORMULARIO_INICIAL)
                
        # Gerar ID único
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
        
        row = [
            id_usuario,
            datetime.now().strftime("%d/%m/%Y às %H:%M"),
            result.get('search_string', ''),
            '',  # termos_sugeridos
            result.get('full_report', '')[:500],
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
            'Sim' if avaliacao_data.get('aceite_continuidade', False) else 'Não',
            ",".join(st.session_state.get('badges', [])),
            tempo_total,
            st.session_state.get('play_video', False),
            st.session_state.get('open_prologo', False)
        ]
        
        worksheet.append_row(row, value_input_option='RAW')
        return True
        
    except Exception as e:
        st.error(f"❌ Erro ao enviar avaliação: {e}")
        return False

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
    """Adiciona badge ao perfil do usuário"""
    if badge_name not in st.session_state.badges:
        st.session_state.badges.append(badge_name)
        return True
    return False

# ==================== ABAS PRINCIPAIS ====================
tab1, tab2 = st.tabs(["📚 Delineascópio", "📊 Painel"])

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
            if '🎯 Explorador' not in st.session_state.badges:
                add_badge('🎯 Explorador')
        else:
            st.info("⏳ 1. Formulário inicial")

    with col2:
        if st.session_state.step >= 2:
            st.success("✅ 2. Grafo de conceitos")
            if '🔬 Pesquisador' not in st.session_state.badges:
                add_badge('🔬 Pesquisador')
        else:
            st.info("⏳ 2. Grafo de conceitos")

    with col3:
        if st.session_state.step >= 2 and sub_step in ['b', 'c']:
            st.success("✅ 3. Seleção de conceitos")
            if '🧩 Seletor' not in st.session_state.badges:
                add_badge('🧩 Seletor')
        elif st.session_state.step == 2 and sub_step == 'a':
            st.info("⏳ 3. Seleção de conceitos")
        else:
            st.info("⏳ 3. Seleção de conceitos")

    with col4:
        if st.session_state.step >= 2 and sub_step == 'c':
            st.success("✅ 4. Relatório")
            if '🏆 Delineador' not in st.session_state.badges:
                add_badge('🏆 Delineador')
        elif st.session_state.step > 2:
            st.success("✅ 4. Relatório")
            if '🏆 Delineador' not in st.session_state.badges:
                add_badge('🏆 Delineador')
        else:
            st.info("⏳ 4. Relatório")

    with col5:
        if st.session_state.get('avaliacao_completa', False):
            st.success("✅ 5. Avaliação")
            if '💎 Avaliador' not in st.session_state.badges:
                add_badge('💎 Avaliador')
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
                        'tema': tema,
                        'questao': questao,
                        'palavras_chave': palavras_chave,
                        'confianca': confianca,
                        'google_academico': google_academico,
                        'timestamp': datetime.now().strftime("%d/%m/%Y às %H:%M")
                    }

                    # Enviar para Google Sheets e salvar ID
                    id_usuario = enviar_formulario_inicial(st.session_state.form_data)
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
                            st.session_state.resultado = pipe.process(nome, tema, questao, kws)
                            tempo_fim = time_module.time()

                            # Enviar resultados para Google Sheets
                            if 'id_usuario' in st.session_state:
                                enviar_resultados_pipeline(
                                    st.session_state.id_usuario,
                                    st.session_state.resultado,
                                    tempo_fim - tempo_inicio
                                )

                            # Avançar para próxima etapa
                            st.session_state.step = 2
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Erro ao processar: {str(e)}")
                            st.exception(e)

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

            st.divider()

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

        # ========== SUB-ETAPA 2b: SELEÇÃO DE CONCEITOS ==========
        elif sub_step == 'b':
            top_concepts = r.get('top_concepts', [])[:9]

            st.header("🎯 3. Seleção de Conceitos")
            st.caption("Etapa 3: Escolha os conceitos mais relevantes para sua pesquisa")

            # Navegação
            if st.button("⬅️ Voltar ao Grafo"):
                st.session_state.sub_step = 'a'
                st.rerun()

            st.divider()

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

            st.divider()

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
            st.divider()

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
                                tema, primeiro_nome, selected, all_concepts
                            )

                            # Gerar sugestões de palavras-chave
                            st.session_state.suggested_keywords = gemini.generate_keyword_suggestions(
                                tema, primeiro_nome, selected, original_kws
                            )

                            # Gerar strings de busca
                            st.session_state.suggested_strings = gemini.generate_search_strings(
                                tema, selected, original_kws
                            )

                            st.session_state.interpretation_generated = True

                        st.session_state.sub_step = 'c'
                        st.rerun()
                else:
                    st.button("Gerar Interpretação Personalizada ▶️", disabled=True, use_container_width=True)

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

            st.divider()

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

            # ========== SEÇÃO 6: STRINGS DE BUSCA SUGERIDAS ==========
            st.subheader("🔎 Strings de Busca Sugeridas")
            st.caption("Copie as strings abaixo para usar no Painel ou em bases de dados")

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
                                st.toast(f"✅ String copiada para o Painel!")
            else:
                # Fallback: mostrar string original
                search_string = r.get('search_string', 'N/A')
                with st.container(border=True):
                    st.markdown("**🔎 String de Busca Original**")
                    col_str, col_btn = st.columns([4, 1])
                    with col_str:
                        st.code(search_string, language='text')
                    with col_btn:
                        if st.button("📋 Copiar", key="copy_original", use_container_width=True):
                            st.session_state.dashboard_query = search_string
                            st.toast("✅ String copiada para o Painel!")

            # ========== SEÇÃO 7: AÇÕES FINAIS ==========
            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                # PDF disponível após completar a trilha
                try:
                    # Adicionar dados da seleção ao resultado para o PDF
                    r_completo = r.copy()
                    r_completo['selected_concepts'] = selected
                    r_completo['personalized_interpretation'] = st.session_state.get('personalized_interpretation', '')
                    r_completo['suggested_keywords'] = st.session_state.get('suggested_keywords', [])
                    r_completo['suggested_strings'] = st.session_state.get('suggested_strings', {})

                    pdf_bytes = generate_pdf_report(d, r_completo)
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
                    st.info("💡 Use as strings sugeridas no Painel para explorar mais a literatura!")

            with col3:
                if st.button("📝 Avaliar Sistema", type="primary", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()

            # Dica final
            st.divider()
            st.info("""
            🎉 **Parabéns!** Você completou a trilha de delineamento!

            Agora você pode:
            - 📥 **Baixar o PDF** com o relatório completo
            - 📊 **Usar o Painel** para explorar mais a literatura
            - 📝 **Avaliar o sistema** e nos ajudar a melhorar
            """)

            # Botão novo projeto
            if st.button("🔄 Iniciar Novo Projeto", use_container_width=True):
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

# ========== ETAPA 3: AVALIAÇÃO EXPANDIDA ==========
    elif st.session_state.step == 3:
        st.header("⭐ 5. Avaliação")
        st.caption("Suas respostas são fundamentais para aprimorarmos o sistema!")

        st.info("""
📊 **Termo de Consentimento Livre e Esclarecido**
 
Convidamos você a participar da pesquisa sobre o uso de palavras-chave na pesquisa acadêmica. Sua participação é totalmente voluntária, e você pode desistir a qualquer momento sem nenhum prejuízo.

O objetivo do estudo é investigar como a avaliação automatizada de definições preliminares de um projeto, como tema, questão de pesquisa e palavras-chave, pode apoiar estudantes no delineamento do escopo do estudo e na delimitação mais precisa de suas propostas.

Ressaltamos que nenhuma informação identificável é utilizada na pesquisa.

Caso tenha dúvidas ou necessite de mais informações, entre em contato por e-mail com o pesquisador responsável, Rafael Antunes dos Santos (rafael.antunes@ufrgs.br), doutorando do Programa de Pós-Graduação em Informática na Educação, da Universidade Federal do Rio Grande do Sul.
                
Ao prosseguir com o preenchimento deste formulário, você declara que entende os objetivos da pesquisa e concorda em participar voluntariamente.
""")

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
                "F2.10. A string oferecida é precisa para o meu tema",
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
                "✅ **Sim, aceito participar de outras fases desta pesquisa e autorizo contato por e-mail**",
                key="aceite_continuidade",
                help="Ao marcar esta opção, você demonstra interesse em contribuir com o desenvolvimento do Delinéia"
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
                    'aceite_continuidade': aceite_continuidade,
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

                # Badge de conclusão
                if '💎 Avaliador' not in st.session_state.badges:
                    add_badge('💎 Avaliador')

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
    
    # ========== ETAPA 4: CONCLUSÃO ==========
    elif st.session_state.step == 4:
        st.success("🎉 Parabéns! Você completou todas as etapas!")
        st.markdown("### 🏆 Conquista Desbloqueada: Delineador!")
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

        st.divider()

        if st.button("🔄 Iniciar Novo Delineamento", use_container_width=True):
            st.session_state.step = 1
            st.session_state.resultado = None
            st.session_state.form_data = {}
            st.session_state.avaliacao_completa = False
            st.session_state.badges = []
            st.rerun()

# ==================== ABA 2: PAINEL DE ANÁLISE ====================
with tab2:
    st.title("📊 Painel de Exploração de Dados")
    st.caption("Análise profunda dos dados do OpenAlex")

    # Sidebar para configuração
    with st.sidebar:
        st.header("⚙️ Configurar Busca")

        # Campo de busca
        query = st.text_input(
            "String de Busca:",
            value=st.session_state.get('dashboard_query', "HIV/AIDS AND Brasil"),
            help="Use operadores: AND, OR, NOT"
        )

        if 'dashboard_query' in st.session_state and st.session_state.dashboard_query:
            st.info("📋 String copiada do Delineascópio")

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
            with st.spinner("🔄 Em processamento, confira no Painel"):
                try:
                    # Inicializar cliente
                    client = OpenAlexClient(OPENALEX_EMAIL)

                    # Buscar artigos
                    articles = client.search_articles(client.normalize_query(query), limit)

                    # Extrair conceitos
                    concepts_lists = []
                    for article in articles:
                        concepts = [
                            c['name'] for c in article.get('concepts', [])
                            if c['score'] >= min_score and c['level'] >= min_level
                        ]
                        if concepts:
                            concepts_lists.append(concepts)

                    # Construir grafo
                    analyzer = CooccurrenceAnalyzer()
                    G = analyzer.build_graph(concepts_lists, min_cooc)

                    # Salvar dados
                    st.session_state.dashboard_data = {
                        'articles': articles,
                        'concepts_lists': concepts_lists,
                        'graph': G
                    }

                    # Mostrar detalhes
                    with st.expander("📋 Detalhes da Busca"):
                        st.write(f"**String enviada:** {query}")
                        st.write(f"**Limite:** {limit}")
                        st.write(f"**Coocorrência mínima:** {min_cooc}")
                        st.write(f"**Filtros:** score≥{min_score}, level≥{min_level}")
                        st.write(f"**Artigos retornados:** {len(articles)}")
                        st.write(f"**Conceitos extraídos:** {len(concepts_lists)}")
                        st.write(f"**Nós no grafo:** {len(G.nodes())}")

                    st.success(f"✅ {len(articles)} artigos | {len(G.nodes())} conceitos")

                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

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
            - **Painel:** Análise profunda de dados do OpenAlex:
              - **Artigos:** Contagens de artigos e links de acesso
              - **Conceitos:** Contagens de conceitos, nuvem de palavras e Lei de Zipf
              - **Coocorrências:** Contagens de associações entre conceitos e matrizes
              - **Grafo:** Visualização interativa
              - **Mapa Temático:** Posição do cluster
              - **Estatísticas:** Resumo breve
              - **Exportação:** Dados em JSON, CSV, GraphML, BibTeX, RIS
        
            ### Tecnologias
            - Python / Streamlit
            - Google Gemini AI 2.5 Pro / Anthropic Claude Opus 4.5
            - OpenAlex API
            - NetworkX, Plotly, ReportLab
        
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

            1. **Digite uma string de busca** (ex: "machine learning AND education")
            2. **Ajuste os filtros** conforme necessário
            3. **Clique em Buscar** para processar
            4. **Explore as abas** com diferentes análises
            5. **Exporte os dados** quando necessário

            **Dica:** Você pode copiar a string de busca do Delineascópio!
            """)

    else:
        # Recuperar dados
        data = st.session_state.dashboard_data
        articles = data['articles']
        concepts_lists = data['concepts_lists']
        G = data['graph']

        # Criar sub-abas para análises
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📚 Artigos",
            "🧩 Conceitos",
            "🔗 Coocorrências",
            "🕸️ Grafo",
            "🗺️ Mapa Temático",
            "📊 Estatísticas",
            "💾 Exportação"
        ])

        # ========== SUB-ABA 1: ARTIGOS (COM DOI/URL) - VERSÃO CORRIGIDA ==========
        with t1:
            st.header("📚 Artigos Analisados")
            st.metric("Total de Artigos", len(articles))

            # ✨ TABELA COM COLUNA DOI/URL ✨
            df_articles = pd.DataFrame([
                {
                    'Título': (lambda t: t[:80] + '...' if len(t) > 80 else t)(a.get('title') or 'Sem título'),
                    'Ano': a.get('year', 'N/A'),
                    'Conceitos': len(a.get('concepts', [])),
                    'DOI/URL': a.get('doi', a.get('url', 'N/A'))
                }
                for a in articles
            ])

            # Configurar coluna como link clicável
            st.dataframe(
                df_articles,
                use_container_width=True,
                height=400,
                column_config={
                    "DOI/URL": st.column_config.LinkColumn(
                        "🔗 DOI/URL",
                        help="Clique para abrir o artigo",
                        display_text="Abrir artigo"
                    )
                }
            )
    
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
                        'Conceito': c['name'],
                        'Score': f"{c['score']:.3f}",
                        'Level': c['level']
                    }
                    for c in selected.get('concepts', [])
                ])
    
                if not concepts_df.empty:
                    st.dataframe(concepts_df, use_container_width=True)
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
            st.divider()
            st.subheader("📋 Tabela de Frequências")
            st.dataframe(df_freq, use_container_width=True)

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

            df_pairs = pd.DataFrame([
                {
                    'Conceito 1': c1,
                    'Conceito 2': c2,
                    'Frequência': f
                }
                for (c1, c2), f in pairs.most_common(top_pairs)
            ])

            st.dataframe(df_pairs, use_container_width=True)

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

        st.divider()

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

        st.divider()

        # ========== SUB-ABA 5: MAPA TEMÁTICO =========
        with t5:
            st.header("🗺️ Mapa Temático (Diagrama Estratégico)")

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
                        from thematic_map_module import ThematicMapAnalyzer

                        with st.spinner("🔄 Detectando clusters e calculando métricas do mapa temático..."):
                            tm_analyzer = ThematicMapAnalyzer(G, concepts_lists)
                            tm_analyzer.detect_clusters(
                                method=cluster_method,
                                min_size=min_cluster_size
                            )
                            metrics_df = tm_analyzer.analyze_clusters()

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

                            # garante alinhamento: mesma ordem de metrics_df e tm_analyzer.clusters
                            for idx, row in metrics_df.reset_index(drop=True).iterrows():
                                quadrante = ThematicMapAnalyzer.classify_quadrant(
                                    row["centralidade_norm"],
                                    row["densidade_norm"]
                                )
                                tipo = tipo_map.get(quadrante, "Basic Theme")

                                # conceitos do cluster (set -> lista ordenada)
                                conceitos_cluster = sorted(tm_analyzer.clusters[idx])
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

        # ========== SUB-ABA 7: EXPORTAÇÃO ==========
        with t7:
            st.header("💾 Exportação de Dados")

            col1, col2, col3 = st.columns(3)

            # JSON
            with col1:
                st.subheader("📄 JSON")

                st.download_button(
                    "📥 Artigos (JSON)",
                    json.dumps(articles, indent=2, ensure_ascii=False),
                    "articles.json",
                    "application/json",
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

            # CSV
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

            # Outros formatos
            with col3:
                st.subheader("🔧 Outros")

                import tempfile

                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.graphml') as f:
                    nx.write_graphml(G, f.name)
                    with open(f.name, 'r') as file:
                        graphml_content = file.read()

                st.download_button(
                    "📥 Grafo (GraphML)",
                    graphml_content,
                    "graph.graphml",
                    "application/xml",
                    use_container_width=True
                )

                st.caption("Para Gephi/Cytoscape")

            st.divider()

            # Zip completo
            st.subheader("📦 Pacote Completo")

            if st.button("🎁 Gerar ZIP com Todos os Dados", use_container_width=True):
                with st.spinner("📦 Gerando arquivo ZIP..."):
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

                        # GraphML
                        zf.writestr('graph.graphml', graphml_content)

                        # README
                        readme = f"""# Delinéia IX - Dados Exportados

Data: {datetime.now().strftime("%d/%m/%Y às %H:%M")}
Query: {query}

## Arquivos incluídos:

### JSON
- articles.json: Lista completa de artigos
- concepts.json: Conceitos extraídos por artigo
- cooccurrences.json: Pares de coocorrências

### CSV
- articles.csv: Artigos (título, ano, num_conceitos)
- concepts.csv: Conceitos e frequências
- cooccurrences.csv: Rede de coocorrências

### Grafo
- graph.graphml: Grafo no formato GraphML (Gephi/Cytoscape)

## Estatísticas:
- Artigos: {len(articles)}
- Conceitos únicos: {len(freq)}
- Nós no grafo: {len(G.nodes())}
- Arestas: {len(G.edges())}
"""
                        zf.writestr('README.txt', readme)

                    st.download_button(
                        "📥 Baixar painel_completo.zip",
                        zip_buffer.getvalue(),
                        "painel_completo.zip",
                        "application/zip",
                        use_container_width=True
                    )