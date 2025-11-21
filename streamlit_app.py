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

# ==================== FUNÇÕES AUXILIARES ====================
def add_badge(badge_name: str) -> bool:
    """Adiciona badge ao perfil do usuário"""
    if badge_name not in st.session_state.badges:
        st.session_state.badges.append(badge_name)
        return True
    return False

# ==================== ABAS PRINCIPAIS ====================
tab1, tab2 = st.tabs(["📚 Delineascópio", "📊 Dashboard"])

# ==================== ABA 1: DELINEASCÓPIO ====================
with tab1:
    st.title("📚 Delinéia - Sistema de Delineamento de Escopo Temático")
    st.caption("Ferramenta de apoio ao delineamento de projetos de pesquisa com IA e bibliometria")

    # Barra de progresso gamificada
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.step >= 1:
            st.success("✅ Etapa 1/3: Formulário")
            if '🎯 Explorador' not in st.session_state.badges:
                add_badge('🎯 Explorador')
        else:
            st.info("⏳ Etapa 1/3: Formulário inicial")

    with col2:
        if st.session_state.step >= 2:
            st.success("✅ Etapa 2/3: Relatório")
            if '🔬 Pesquisador' not in st.session_state.badges:
                add_badge('🔬 Pesquisador')
        else:
            st.info("⏳ Etapa 2/3: Aguardando dados")

    with col3:
        if st.session_state.step >= 3:
            st.success("✅ Etapa 3/3: Avaliação")
            if '🏆 Mestre Delineador' not in st.session_state.badges:
                add_badge('🏆 Mestre Delineador')
        else:
            st.info("⏳ Etapa 3/3: Avaliação")

    # Mostrar badges conquistados
    if st.session_state.badges:
        st.markdown(f"**🏅 Conquistas:** {' '.join(st.session_state.badges)}")

    st.divider()

    # ========== ETAPA 1: FORMULÁRIO INICIAL ==========
    if st.session_state.step == 1:
        st.header("📝 Formulário Inicial")

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
                placeholder="Ex: Pesquisa brasileira em HIV/AIDS",
                help="Tema principal do seu projeto"
            )

            questao = st.text_area(
                "F1.2. Questão de Pesquisa*",
                placeholder="Ex: Como os tópicos de pesquisa em HIV/AIDS evoluíram no Brasil?",
                height=100,
                help="Pergunta principal que você quer responder"
            )

            palavras_chave = st.text_input(
                "F1.3. Palavras-chave*",
                placeholder="Ex: HIV/AIDS, Pesquisa, Brasil",
                help="Separe as palavras-chave por vírgula"
            )

            google_academico = st.text_area(
                "F1.4. Se você fosse pesquisar referências para seu projeto no Google Acadêmico, o que você colocaria no campo de busca?*",
                placeholder="Ex: Pesquisas sobre HIV/AIDS no Brasil",
                help="Campo livre para indicar palavras, frases, etc. que você quer pesquisar",
                height=100
            )

            st.divider()
            st.subheader("💭 Autoavaliação")

            confianca = st.select_slider(
                "F1.5. Qual seu nível de segurança em relação às palavras-chave escolhidas?",
                options=[
                    "Totalmente inseguro",
                    "Inseguro",
                    "Neutro",
                    "Seguro",
                    "Totalmente seguro"
                ],
                value="Neutro",
                help="Avalie sua confiança nas palavras-chave selecionadas"
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

                    with st.spinner("🔄 Processando... (aguarde 2-3 minutos)"):
                        try:
                            # Inicializar pipeline
                            pipe = ResearchScopePipeline(OPENALEX_EMAIL)

                            # Processar palavras-chave
                            kws = [k.strip() for k in palavras_chave.split(',') if k.strip()]

                            # Executar pipeline
                            st.session_state.resultado = pipe.process(nome, tema, questao, kws)

                            # Avançar para próxima etapa
                            st.session_state.step = 2
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Erro ao processar: {str(e)}")
                            st.exception(e)

    # ========== ETAPA 2: RELATÓRIO ==========
    elif st.session_state.step == 2:
        d = st.session_state.form_data
        r = st.session_state.resultado

        st.header("📊 Relatório de Delineamento")
        st.caption("[ Relatório produzido por Inteligência Artificial ]")

        # Botões de ação
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            if st.button("⬅️ Voltar ao Formulário"):
                st.session_state.step = 1
                st.rerun()

        with col2:
            try:
                pdf_bytes = generate_pdf_report(d, r)
                st.download_button(
                    "📥 Baixar PDF",
                    pdf_bytes,
                    f"delineamento_{d['nome'].replace(' ', '_')}.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Erro ao gerar PDF: {str(e)}")

        with col3:
            if st.button("📝 Avaliar Sistema", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()

        with col4:
            if st.button("🔄 Novo Projeto"):
                st.session_state.step = 1
                st.session_state.resultado = None
                st.session_state.form_data = {}
                st.session_state.avaliacao_completa = False
                st.session_state.badges = []
                st.rerun()

        st.divider()

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

        # Avaliação do projeto
        st.subheader("📋 Avaliação do Projeto")
        st.markdown(r.get('full_report', '⚠️ Avaliação não disponível'))

        st.markdown("Com base nas palavras-chave fornecidas, desenvolvemos uma string de busca:")

        # String de busca
        st.subheader("🔎 String de Busca")

        search_string = r.get('search_string', 'N/A')

        col_a, col_b = st.columns([3, 1])

        with col_a:
            with st.expander("📄 Ver string completa", expanded=True):
                st.code(search_string, language='text')

        with col_b:
            if st.button("📋 Copiar para Dashboard", use_container_width=True):
                st.session_state.dashboard_query = search_string
                st.success("✅ Copiado!")

        st.write(f"**Objetivo:** {r.get('search_objective', '')}")

        st.markdown("Elaborou-se um modelo de visualização das coocorrências entre conceitos:")

        # Métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("📚 Artigos Analisados", r.get('articles_count', 0))
        col2.metric("🧩 Conceitos Identificados", r['graph_stats']['nodes'])
        col3.metric("🔗 Conexões no Grafo", r['graph_stats']['edges'])

        # Grafo de coocorrência
        st.subheader("🕸️ Grafo de Coocorrência de Conceitos")

        if r.get('visualization_path'):
            st.image(r['visualization_path'], use_container_width=True)
        else:
            st.warning("⚠️ Visualização não disponível")

        # Glossário
        st.subheader("📖 Glossário de Conceitos")
        st.markdown(r.get('glossary', '⚠️ Glossário não disponível'))

        # Interpretação
        st.subheader("💡 Interpretação do Grafo")
        st.write(r.get('graph_interpretation', '⚠️ Interpretação não disponível'))

        # CTA para avaliação
        st.divider()
        st.info("💝 Ajude a melhorar o Delinéia! Complete a avaliação e desbloqueie o distintivo **🏆 Mestre Delineador**")

        if st.button("➡️ Ir para Avaliação", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ========== ETAPA 3: AVALIAÇÃO EXPANDIDA ==========
    elif st.session_state.step == 3:
        st.header("📋 Avaliação do Sistema Delinéia")
        st.caption("Suas respostas são fundamentais para aprimorarmos a ferramenta!")

        st.info("📊 **Novo formulário expandido:** 30 perguntas + NPS + 4 campos abertos")

        with st.form("formulario_avaliacao"):

            # ==================== SEÇÃO 1: UTILIDADE PERCEBIDA ====================
            st.subheader("💼 Utilidade Percebida")

            q1 = st.radio(
                "1. Usar o Delinéia melhora minha capacidade de delinear o escopo da pesquisa",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q1"
            )

            q2 = st.radio(
                "2. Usar o Delinéia aumenta minha produtividade na definição do projeto",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q2"
            )

            q3 = st.radio(
                "3. O Delinéia é útil para delimitar meu projeto de pesquisa",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q3"
            )

            # ⭐ NOVA PERGUNTA
            q4 = st.radio(
                "4. O Delinéia me ajuda a identificar lacunas na literatura do meu tema",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q4"
            )

            st.divider()

            # ==================== SEÇÃO 2: FACILIDADE DE USO ====================
            st.subheader("🎯 Facilidade de Uso Percebida")

            q5 = st.radio(
                "5. Aprender a usar o Delinéia é fácil para mim",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q5"
            )

            q6 = st.radio(
                "6. A interação com o Delinéia é clara e compreensível",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q6"
            )

            q7 = st.radio(
                "7. Eu acho o Delinéia fácil de usar",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q7"
            )

            # ⭐ NOVA PERGUNTA
            q8 = st.radio(
                "8. A navegação entre as diferentes funcionalidades é intuitiva",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q8"
            )

            st.divider()

            # ==================== SEÇÃO 3: QUALIDADE DA INFORMAÇÃO ====================
            st.subheader("📊 Qualidade da Informação")

            q9 = st.radio(
                "9. As informações fornecidas pelo Delinéia são precisas",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q9"
            )

            q10 = st.radio(
                "10. As análises e sugestões do sistema são relevantes para meu projeto",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q10"
            )

            q11 = st.radio(
                "11. O grafo de coocorrências me ajudou a visualizar relações entre conceitos",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q11"
            )

            # ⭐ NOVA PERGUNTA
            q12 = st.radio(
                "12. A avaliação gerada pela IA é construtiva e específica para meu projeto",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q12"
            )

            st.divider()

            # ==================== ⭐ SEÇÃO 4: FUNCIONALIDADES ESPECÍFICAS (NOVA) ====================
            st.subheader("🗺️ Avaliação de Funcionalidades Específicas")

            q13 = st.radio(
                "13. O Mapa Temático Estratégico me ajudou a posicionar meu tema na literatura",
                ["Não usei", "Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=False,
                key="q13"
            )

            q14 = st.radio(
                "14. A análise de Zipf foi útil para entender a distribuição de conceitos",
                ["Não usei", "Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=False,
                key="q14"
            )

            q15 = st.radio(
                "15. O Dashboard exploratório permite insights que eu não teria manualmente",
                ["Não usei", "Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=False,
                key="q15"
            )

            q16 = st.radio(
                "16. O relatório em PDF é adequado para apresentar ao meu orientador",
                ["Não gerei PDF", "Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=False,
                key="q16"
            )

            st.divider()

            # ==================== ⭐ SEÇÃO 5: COMPARAÇÃO (NOVA) ====================
            st.subheader("⚖️ Comparação com Métodos Tradicionais")

            q17 = st.radio(
                "17. O Delinéia é mais eficiente que realizar buscas manuais em bases de dados",
                ["Nunca fiz busca manual", "Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=False,
                key="q17"
            )

            q18 = st.radio(
                "18. As visualizações do Delinéia são mais informativas que tabelas tradicionais",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q18"
            )

            q19 = st.radio(
                "19. O tempo gasto usando o Delinéia compensa os resultados obtidos",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q19"
            )

            st.divider()

            # ==================== SEÇÃO 6: INTENÇÃO DE USO ====================
            st.subheader("🔮 Intenção de Uso")

            q20 = st.radio(
                "20. Eu pretendo usar o Delinéia em projetos futuros",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q20"
            )

            q21 = st.radio(
                "21. Eu recomendaria o Delinéia para colegas",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q21"
            )

            # ⭐ NOVA PERGUNTA
            q22 = st.radio(
                "22. Eu usaria o Delinéia em diferentes fases da minha pesquisa (projeto, qualificação, defesa)",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q22"
            )

            st.divider()

            # ==================== SEÇÃO 7: CONFIANÇA NO SISTEMA ====================
            st.subheader("🔒 Confiança no Sistema")

            q23 = st.radio(
                "23. Eu confio nas análises geradas pelo Delinéia",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q23"
            )

            q24 = st.radio(
                "24. O sistema demonstra conhecimento sobre metodologia de pesquisa",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q24"
            )

            # ⭐ NOVA PERGUNTA
            q25 = st.radio(
                "25. Eu me sinto confortável em basear decisões acadêmicas nos resultados do Delinéia",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q25"
            )

            st.divider()

            # ==================== SEÇÃO 8: EXPERIÊNCIA DO USUÁRIO ====================
            st.subheader("✨ Experiência do Usuário")

            q26 = st.radio(
                "26. O design da interface é agradável",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q26"
            )

            q27 = st.radio(
                "27. O tempo de processamento do relatório foi adequado",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q27"
            )

            st.divider()

            # ==================== SEÇÃO 9: IMPACTO EDUCACIONAL ====================
            st.subheader("📚 Impacto Educacional")

            q28 = st.radio(
                "28. O Delinéia me ensinou conceitos novos sobre bibliometria",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q28"
            )

            q29 = st.radio(
                "29. Usar o Delinéia melhorou minha alfabetização informacional",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q29"
            )

            # ⭐ NOVA PERGUNTA
            q30 = st.radio(
                "30. O sistema me ajudou a formular perguntas de pesquisa mais precisas",
                ["Discordo Totalmente", "Discordo", "Neutro", "Concordo", "Concordo Totalmente"],
                horizontal=True,
                key="q30"
            )

            st.divider()

            # ==================== ⭐ SEÇÃO 10: NPS (NOVA) ====================
            st.subheader("⭐ Satisfação Geral (Net Promoter Score)")

            nps = st.slider(
                "Em uma escala de 0 a 10, quanto você recomendaria o Delinéia para um colega?",
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
                st.warning("😞 **Detrator** - Queremos ouvir suas sugestões!")

            st.divider()

            # ==================== CAMPOS ABERTOS ====================
            st.subheader("💬 Comentários Adicionais")

            positivo = st.text_area(
                "O que você mais gostou no Delinéia?",
                height=100,
                key="positive_feedback",
                placeholder="Descreva os aspectos mais positivos da sua experiência..."
            )

            melhorias = st.text_area(
                "O que poderia ser melhorado?",
                height=100,
                key="improvements",
                placeholder="Sugestões de melhorias, funcionalidades ausentes, problemas encontrados..."
            )

            futuro = st.text_area(
                "Funcionalidades que você gostaria de ver no futuro:",
                height=100,
                key="future_features",
                placeholder="Ideias para próximas versões..."
            )

            # ⭐ NOVA PERGUNTA ABERTA
            uso = st.text_area(
                "Como você usou (ou pretende usar) os resultados do Delinéia em sua pesquisa?",
                height=100,
                key="usage_context",
                placeholder="Ex: projeto de qualificação, artigo, revisão de literatura..."
            )

            st.divider()

            # ==================== ⭐ DADOS DEMOGRÁFICOS (NOVO) ====================
            st.subheader("👤 Perfil do Respondente (Opcional)")

            col1, col2 = st.columns(2)

            with col1:
                nivel = st.selectbox(
                    "Nível acadêmico:",
                    ["Prefiro não informar", "Graduação", "Especialização", "Mestrado",
                     "Doutorado", "Pós-Doutorado", "Docente"],
                    key="nivel_academico"
                )

                experiencia = st.selectbox(
                    "Experiência prévia com bibliometria:",
                    ["Nenhuma", "Básica", "Intermediária", "Avançada"],
                    key="experiencia_biblio"
                )

            with col2:
                area = st.selectbox(
                    "Área do conhecimento:",
                    ["Prefiro não informar", "Ciências Exatas", "Ciências Biológicas", "Ciências da Saúde",
                     "Ciências Agrárias", "Ciências Sociais Aplicadas", "Ciências Humanas",
                     "Linguística/Letras/Artes", "Engenharias", "Multidisciplinar"],
                    key="area_conhecimento"
                )

                tempo_uso = st.selectbox(
                    "Tempo gasto usando o Delinéia hoje:",
                    ["< 15 min", "15-30 min", "30-60 min", "> 1 hora"],
                    key="tempo_uso"
                )

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
                    # Perguntas Likert
                    'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5,
                    'q6': q6, 'q7': q7, 'q8': q8, 'q9': q9, 'q10': q10,
                    'q11': q11, 'q12': q12, 'q13': q13, 'q14': q14, 'q15': q15,
                    'q16': q16, 'q17': q17, 'q18': q18, 'q19': q19, 'q20': q20,
                    'q21': q21, 'q22': q22, 'q23': q23, 'q24': q24, 'q25': q25,
                    'q26': q26, 'q27': q27, 'q28': q28, 'q29': q29, 'q30': q30,
                    # NPS
                    'nps': nps,
                    'nps_category': nps_category,
                    # Campos abertos
                    'positivo': positivo,
                    'melhorias': melhorias,
                    'futuro': futuro,
                    'uso': uso,
                    # Demográfico
                    'nivel': nivel,
                    'experiencia': experiencia,
                    'area': area,
                    'tempo_uso': tempo_uso,
                    # Metadados
                    'timestamp': datetime.now().isoformat()
                }

                # Salvar em session_state
                st.session_state.avaliacao_completa = True
                st.session_state.avaliacao_data = avaliacao_data

                # Badge de conclusão
                add_badge('🏆 Mestre Avaliador')

                # Feedback visual
                st.success("✅ Avaliação enviada com sucesso!")
                st.balloons()

                # Resumo da avaliação
                st.info(f"""
                📊 **Resumo da sua avaliação:**

                • **NPS:** {nps}/10 ({nps_category})
                • **Nível acadêmico:** {nivel}
                • **Experiência bibliométrica:** {experiencia}
                • **Área:** {area}
                • **Tempo de uso:** {tempo_uso}

                🏆 **Badge desbloqueado:** Mestre Avaliador

                Obrigado por dedicar seu tempo para avaliar o Delinéia!
                Seu feedback é essencial para o desenvolvimento contínuo do sistema.
                """)

                # Avançar para próxima etapa
                st.session_state.step = 4
                st.rerun()

    # ========== ETAPA 4: CONCLUSÃO ==========
    elif st.session_state.step == 4:
        st.success("🎉 Parabéns! Você completou todas as etapas!")
        st.markdown("### 🏆 Conquista Desbloqueada: Mestre Delineador!")
        st.balloons()

        primeiro_nome = st.session_state.form_data['nome'].split()[0]

        st.write(f"**{primeiro_nome}**, você concluiu com sucesso:")
        st.write("✅ Delineamento completo do projeto")
        st.write("✅ Análise bibliométrica avançada")
        st.write("✅ Avaliação do sistema Delinéia")
        st.write(f"\n**🏅 Suas conquistas:** {' '.join(st.session_state.badges)}")

        st.divider()

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("📜 Baixar Certificado de Conclusão", use_container_width=True):
                st.info("🚧 Funcionalidade em desenvolvimento")

        st.divider()

        if st.button("🔄 Iniciar Novo Delineamento", use_container_width=True):
            st.session_state.step = 1
            st.session_state.resultado = None
            st.session_state.form_data = {}
            st.session_state.avaliacao_completa = False
            st.session_state.badges = []
            st.rerun()

# ==================== ABA 2: DASHBOARD DE ANÁLISE ====================
with tab2:
    st.title("📊 Dashboard de Exploração de Dados")
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
                limit = st.slider("Limite de artigos:", 10, 500, 100, 10)
                min_score = st.slider("Score mínimo:", 0.0, 1.0, 0.35, 0.05)
                min_level = st.slider("Level mínimo:", 0, 5, 0, 1)

        min_cooc = st.slider("Coocorrência mínima:", 1, 10, 2, 1)

        st.divider()

        # Botão de buscar
        if st.button("🔍 Buscar", type="primary", use_container_width=True):
            with st.spinner("🔄 Em processamento, confira no Dashboard"):
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
                        st.write(f"**Filtros:** score≥{min_score}, level≥{min_level}")
                        st.write(f"**Artigos retornados:** {len(articles)}")
                        st.write(f"**Conceitos extraídos:** {len(concepts_lists)}")
                        st.write(f"**Nós no grafo:** {len(G.nodes())}")

                    st.success(f"✅ {len(articles)} artigos | {len(G.nodes())} conceitos")

                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

        st.divider()

        # ========== SEÇÃO SOBRE ==========
        with st.expander("📚 Sobre o Delinéia"):
            st.markdown("""
            ### O que é o Delinéia?
            O Delinéia é um sistema de apoio ao delineamento do escopo temático de projetos de pesquisa no ensino superior, desenvolvido como parte de uma tese de doutorado. A ferramenta combina Inteligência Artificial (Google Gemini) com análise bibliométrica de coocorrência de palavras (OpenAlex) para auxiliar estudantes de graduação e de pós-graduação no esboço de seus projetos de pesquisa.
        
            ### Desenvolvimento
            **Autor:** Rafael Antunes dos Santos  
            **Instituição:** Universidade Federal do Rio Grande do Sul (UFRGS), Centro Interdisciplinar de Novas Tecnologias na Educação (Cinted), Programa de Pós-Graduação em Informática na Educação (PPGIE)  
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
            - **Dashboard:** Análise profunda dos dados do OpenAlex:
              - **Artigos:** Contagens de artigos e links de acesso
              - **Conceitos:** Contagens de conceitos e Lei de Zipf
              - **Coocorrências:** Contagens de associações entre conceitos e matriz de calor
              - **Grafo:** Visualização interativa
              - **Mapa Temático:** Posição de cluster na literatura recuperada
              - **Estatísticas:** Resumo breve
              - **Exportação:** Dados em JSON, CSV, GraphML
        
            ### Tecnologias
            - Python / Streamlit
            - Google Gemini AI 2.5 Pro / Anthropic Claude Sonnet 4.5
            - OpenAlex API
            - NetworkX, Plotly, ReportLab
        
            ### Contato
            📧 rafael.antunes@ufrgs.br
        
            ### Versão
            Delinéia XIV - 2025

            ### Agradecimentos
            Aos Professores Leandro Krug Wives, Rosa Maria Vicari, Dante Augusto Couto Barone, Patrícia Fernanda da Silva, Sérgio Roberto Kieling Franco, Renato Ventura Bayan Henriques, Milton Antonio Zaro, Fernando Becker, Vanessa Soares Maurente, Elisa Boff, Alessandra Lorandi e Gabriela Trindade Perry
            Aos colegas do grupo de pesquisa GTech.Edu
            À CAPES
            """)
    
    # Área principal do dashboard
    if st.session_state.dashboard_data is None:
        st.info("👈 Configure os parâmetros na barra lateral e clique em **Buscar** para iniciar a análise")

        # Mostrar exemplo
        with st.expander("💡 Exemplo de uso"):
            st.markdown("""
            **Como usar o Dashboard:**

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

        # ========== SUB-ABA 1: ARTIGOS (COM DOI/URL) ==========
        with t1:
            st.header("📚 Artigos Analisados")
            st.metric("Total de Artigos", len(articles))

            # ✨ TABELA COM COLUNA DOI/URL ✨
            df_articles = pd.DataFrame([
                {
                    'Título': a.get('title', '')[:80] + '...' if len(a.get('title', '')) > 80 else a.get('title', ''),
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

                # Seletor de artigo
                idx = st.selectbox(
                    "Selecione um artigo:",
                    range(len(articles)),
                    format_func=lambda i: f"{i+1}. {articles[i].get('title', '')[:60]}..."
                )

                selected = articles[idx]

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Título:** {selected.get('title', 'N/A')}")
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

            - 🎯 **Motor Themes**: Centrais e bem desenvolvidos (PRIORIZE)
            - 🔷 **Niche Themes**: Especializados e coesos
            - 🔶 **Basic Themes**: Transversais, mas em desenvolvimento
            - 🔴 **Emerging/Declining**: Fronteiras de pesquisa
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
                                    "Niche Theme": "🔷",
                                    "Emerging/Declining Theme": "🔴",
                                    "Basic Theme": "🔶",
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
                                        if cluster["tipo"] == "Motor Theme":
                                            st.success("💡 Tema central e maduro. **PRIORIZE** na revisão de literatura.")
                                        elif cluster["tipo"] == "Niche Theme":
                                            st.info(f"💡 Tema especializado. Útil para nichos relacionados a '{cluster['conceito_principal']}'.")
                                        elif cluster["tipo"] == "Basic Theme":
                                            st.warning("💡 Tema transversal. Oportunidade para pesquisas integradoras.")
                                        else:
                                            st.error("💡 Tema emergente ou em declínio. Fronteira de pesquisa.")

                            # ---------- Explicação metodológica ----------
                            with st.expander("ℹ️ Sobre a metodologia (Aria & Cuccurullo, 2017; He, 1999)"):
                                st.markdown("""
                                Este mapa temático segue a lógica do *Strategic Diagram*:

                                - **Densidade**: média dos pesos das arestas internas do cluster (coesão interna).
                                - **Centralidade**: soma dos pesos das arestas que ligam o cluster a outros clusters (relevância global).
                                - A posição de cada cluster no plano Centralidade × Densidade permite interpretar seu papel
                                  na estrutura do campo de pesquisa.

                                Referências:

                                - Aria, M., & Cuccurullo, C. (2017). *bibliometrix: An R-tool for comprehensive science mapping analysis.*
                                  Journal of Informetrics, 11(4), 959–975.

                                - He, Q. (1999). *Knowledge discovery through co-word analysis.*
                                  Library Trends, 48(1), 133–159.
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
                        "📥 Baixar dashboard_completo.zip",
                        zip_buffer.getvalue(),
                        "dashboard_completo.zip",
                        "application/zip",
                        use_container_width=True
                    )
