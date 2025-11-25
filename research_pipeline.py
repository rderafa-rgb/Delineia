# -*- coding: utf-8 -*-
"""
RESEARCH PIPELINE - VERSÃO DIAGNÓSTICO
======================================
Esta versão mostra VISUALMENTE no Streamlit o que está acontecendo
para identificar EXATAMENTE onde o Gemini está falhando.
"""

import time
import re
from typing import List, Dict, Tuple
import requests
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

# ==================== FUNÇÕES DE DIAGNÓSTICO ====================
def log_diagnostico(mensagem: str, tipo: str = "info"):
    """Mostra mensagem de diagnóstico no Streamlit"""
    if tipo == "success":
        st.success(f"✅ {mensagem}")
    elif tipo == "error":
        st.error(f"❌ {mensagem}")
    elif tipo == "warning":
        st.warning(f"⚠️ {mensagem}")
    else:
        st.info(f"ℹ️ {mensagem}")
    print(f"[{tipo.upper()}] {mensagem}")


# ==================== CLIENTE OPENALEX ====================
class OpenAlexClient:
    """Cliente para buscar artigos no OpenAlex"""

    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://api.openalex.org/works"

    def normalize_query(self, query: str) -> str:
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    def search_articles(self, query: str, limit: int = 500) -> List[Dict]:
        results = []
        
        for page in range(1, 4):
            params = {
                'search': query,
                'per_page': min(200, limit - len(results)),
                'page': page,
                'mailto': self.email
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    works = data.get('results', [])
                    
                    for work in works:
                        doi = work.get('doi', '')
                        if doi and doi.startswith('https://doi.org/'):
                            doi = doi
                        elif doi:
                            doi = f"https://doi.org/{doi}"
                        
                        results.append({
                            'id': work.get('id', ''),
                            'title': work.get('title', ''),
                            'year': work.get('publication_year', ''),
                            'doi': work.get('doi', ''),
                            'url': doi if doi else work.get('id', ''),
                            'concepts': [
                                {'name': c.get('display_name', ''), 
                                 'score': c.get('score', 0),
                                 'level': c.get('level', 0)}
                                for c in work.get('concepts', [])
                            ]
                        })
                    
                    if len(results) >= limit or len(works) < 200:
                        break
                else:
                    break
                    
            except Exception as e:
                break
        
        return results[:limit]

    def extract_concepts_for_cooccurrence(self, articles: List[Dict],
                                         min_score: float = 0.35,
                                         min_level: int = 0) -> List[List[str]]:
        concepts_lists = []
        
        for article in articles:
            concepts = [
                c['name']
                for c in article.get('concepts', [])
                if c.get('score', 0) >= min_score and c.get('level', 0) >= min_level
            ]
            
            if concepts:
                concepts_lists.append(concepts)
        
        return concepts_lists


# ==================== GERADOR GEMINI COM DIAGNÓSTICO ====================
class GeminiQueryGenerator:
    """
    Gerador de análises usando Gemini AI.
    VERSÃO COM DIAGNÓSTICO VISUAL
    """

    def __init__(self):
        self.model = None
        self.api_key_status = "não verificada"
        
        try:
            # DIAGNÓSTICO 1: Verificar API Key
            api_key = st.secrets.get("GEMINI_API_KEY", "")
            
            if not api_key:
                self.api_key_status = "VAZIA ou NÃO ENCONTRADA"
                log_diagnostico(f"GEMINI_API_KEY: {self.api_key_status}", "error")
                return
            
            self.api_key_status = f"encontrada ({len(api_key)} chars, começa com {api_key[:10]}...)"
            log_diagnostico(f"GEMINI_API_KEY: {self.api_key_status}", "success")
            
            # DIAGNÓSTICO 2: Configurar API
            genai.configure(api_key=api_key)
            log_diagnostico("genai.configure() executado", "success")
            
            # DIAGNÓSTICO 3: Criar modelo
            self.model = genai.GenerativeModel(
                'gemini-2.5-pro',
                generation_config={
                    'temperature': 1.2,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 8192,
                }
            )
            log_diagnostico(f"Modelo criado: {self.model.model_name}", "success")
            
        except Exception as e:
            log_diagnostico(f"ERRO na inicialização: {type(e).__name__}: {str(e)}", "error")
            self.model = None

    def _safe_generate(self, prompt: str, fallback: str = "", max_retries: int = 3) -> str:
        """Geração segura com DIAGNÓSTICO COMPLETO"""
        
        # DIAGNÓSTICO: Verificar se modelo existe
        if not self.model:
            log_diagnostico("Modelo não disponível - usando FALLBACK", "error")
            return fallback

        log_diagnostico(f"Iniciando geração (prompt: {len(prompt)} chars)", "info")

        for attempt in range(max_retries):
            try:
                log_diagnostico(f"Tentativa {attempt + 1}/{max_retries}...", "info")
                
                # DIAGNÓSTICO: Medir tempo
                start_time = time.time()
                response = self.model.generate_content(prompt)
                elapsed = time.time() - start_time
                
                log_diagnostico(f"Resposta recebida em {elapsed:.2f}s", "success")
                
                # DIAGNÓSTICO: Verificar prompt_feedback (bloqueio)
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    log_diagnostico(f"Prompt feedback: {feedback}", "info")
                    if hasattr(feedback, 'block_reason') and feedback.block_reason:
                        log_diagnostico(f"BLOQUEADO: {feedback.block_reason}", "error")
                        continue

                extracted_text = None

                # Método 1: .text
                if hasattr(response, 'text'):
                    try:
                        extracted_text = response.text
                        log_diagnostico(f"Método .text: {len(extracted_text) if extracted_text else 0} chars", "success")
                    except ValueError as ve:
                        log_diagnostico(f"Método .text falhou: {ve}", "warning")

                # Método 2: candidates
                if not extracted_text and hasattr(response, 'candidates') and response.candidates:
                    try:
                        candidate = response.candidates[0]
                        
                        if hasattr(candidate, 'finish_reason'):
                            log_diagnostico(f"Finish reason: {candidate.finish_reason}", "info")
                        
                        if hasattr(candidate, 'safety_ratings'):
                            for rating in candidate.safety_ratings:
                                prob = str(getattr(rating, 'probability', 'N/A'))
                                if prob in ['HIGH', 'MEDIUM']:
                                    cat = str(getattr(rating, 'category', 'N/A'))
                                    log_diagnostico(f"Safety: {cat} = {prob}", "warning")
                        
                        if hasattr(candidate, 'content') and candidate.content:
                            parts = candidate.content.parts
                            if parts and len(parts) > 0:
                                extracted_text = parts[0].text
                                log_diagnostico(f"Método candidates: {len(extracted_text)} chars", "success")
                    except Exception as ce:
                        log_diagnostico(f"Método candidates falhou: {ce}", "warning")

                # Método 3: str()
                if not extracted_text:
                    raw_str = str(response)
                    log_diagnostico(f"Método str(): {len(raw_str)} chars - '{raw_str[:100]}...'", "warning")
                    if len(raw_str) > 50 and "response:" not in raw_str.lower():
                        extracted_text = raw_str

                # Validação final
                if extracted_text:
                    extracted_text = extracted_text.strip()
                    
                    if len(extracted_text) >= 30 and extracted_text != "None":
                        log_diagnostico(f"SUCESSO! Texto válido: {len(extracted_text)} chars", "success")
                        log_diagnostico(f"Preview: {extracted_text[:150]}...", "info")
                        return extracted_text
                    else:
                        log_diagnostico(f"Texto muito curto/inválido: {len(extracted_text)} chars", "warning")

                if attempt < max_retries - 1:
                    log_diagnostico("Aguardando 3s antes de retry...", "info")
                    time.sleep(3)

            except Exception as e:
                log_diagnostico(f"EXCEÇÃO: {type(e).__name__}: {str(e)[:200]}", "error")
                if attempt < max_retries - 1:
                    time.sleep(3)

        log_diagnostico("USANDO FALLBACK após todas as tentativas", "error")
        return fallback

    def generate_full_report(self, nome: str, tema: str, questao: str,
                            keywords: List[str]) -> str:
        """Gera avaliação crítica e construtiva do projeto"""
        keywords_str = ', '.join(keywords)
        primeiro_nome = nome.split()[0] if nome else "estudante"

        prompt = f"""Você é um professor universitário experiente orientando um aluno de pesquisa.

**CONTEXTO DO PROJETO:**

Aluno: {nome} (você vai chamá-lo de {primeiro_nome})
Tema proposto: {tema}
Questão de pesquisa: {questao}
Palavras-chave escolhidas: {keywords_str}

---

**SUA TAREFA:**

Escreva DOIS parágrafos conversando com {primeiro_nome}:

**PARÁGRAFO 1 - Sobre as palavras-chave:**
• Comece com: "{primeiro_nome}, as palavras-chave que você designou para o projeto..."
• Comente especificamente sobre as palavras-chave escolhidas
• Seja autêntico: se estão boas, diga o que está bom; se há problemas, aponte com clareza mas cuidado
• Se palavras forem muito amplas, diga quais e por quê
• Se houver redundância, mostre
• Se faltar algo importante, sugira especificamente

**PARÁGRAFO 2 - Sobre a questão de pesquisa:**
• Comente explicitamente sobre a questão de pesquisa apresentada
• Analise se está clara, viável e bem delimitada
• Sugira refinamentos se necessário
• Relacione com as palavras-chave escolhidas
• Encerre com: "Recomendo que você converse com seu orientador sobre esses pontos e observe atentamente o grafo de coocorrências apresentado adiante, pois ele pode revelar relações importantes entre conceitos que ajudarão a refinar suas palavras-chave e a delimitar melhor o escopo da sua pesquisa."

**DIRETRIZES:**
• Tom de conversa: use "você" e o primeiro nome
• Honesto mas respeitoso
• Como um professor que realmente se importa com o aluno
• NÃO use linguagem de parecer formal
• Seja específico sobre ESTAS palavras-chave e ESTA questão
• NÃO use frases genéricas que servem para qualquer projeto
• Projetos absurdos ou inviáveis merecem feedback honesto

**IMPORTANTE:** NÃO use frases como "Com certeza..." ou expressões clichê. Seja direto e genuíno.

---

Escreva agora os dois parágrafos para {primeiro_nome}:"""

        fallback = f"""{primeiro_nome}, as palavras-chave que você designou para o projeto ({keywords_str}) cobrem alguns aspectos do tema '{tema}'. No entanto, seria importante avaliar se esses termos capturam as nuances específicas da sua questão de pesquisa e se há necessidade de termos mais específicos ou complementares.

Sobre sua questão de pesquisa, '{questao}', é fundamental verificar se está suficientemente delimitada e se oferece um caminho claro para investigação. Recomendo que você converse com seu orientador sobre esses pontos e observe atentamente o grafo de coocorrências apresentado adiante, pois ele pode revelar relações importantes entre conceitos que ajudarão a refinar suas palavras-chave e a delimitar melhor o escopo da sua pesquisa."""

        return self._safe_generate(prompt, fallback)

    def generate_suggested_keywords(self, nome: str, tema: str, questao: str,
                                   keywords: List[str]) -> str:
        """Sugere palavras-chave complementares em inglês técnico"""
        keywords_str = ', '.join(keywords)

        prompt = f"""Você é um bibliotecário especializado em buscas científicas.

**PROJETO:**
Tema: {tema}
Questão: {questao}
Palavras atuais do aluno: {keywords_str}

**TAREFA:**
Liste 4-6 termos técnicos EM INGLÊS que sejam:
- Complementares (NÃO repetir os que o aluno já tem)
- Específicos da área de pesquisa
- Reconhecidos na literatura científica internacional
- Úteis para ampliar a busca mantendo relevância

**INSTRUÇÕES:**
- Retorne APENAS os termos separados por vírgula
- Sem numeração, sem aspas, sem explicações
- Apenas: termo1, termo2, termo3, termo4

**EXEMPLO do formato correto:**
cognitive load, metacognition, learning strategies, self-regulation

Gere agora os termos complementares:"""

        fallback = "research methods, empirical studies, theoretical framework, scientific literature"

        result = self._safe_generate(prompt, fallback)
        result = result.replace('\n', ', ').strip()
        return result

    def translate_keywords_to_english(self, keywords: List[str]) -> List[str]:
        """Traduz palavras-chave do português para inglês."""
        keywords_str = ', '.join(keywords)

        prompt = f"""Você é um tradutor especializado em terminologia científica.

**TAREFA:**
Traduza os seguintes termos do PORTUGUÊS para INGLÊS acadêmico/técnico.

**TERMOS:**
{keywords_str}

**INSTRUÇÕES:**
- Retorne APENAS os termos traduzidos
- Mesma ordem do original
- Separados por vírgula
- Use terminologia padrão em publicações científicas
- Sem numeração, sem explicações

**EXEMPLO:**
Entrada: Psicologia, Escola, Professores, Burnout
Saída: Psychology, School, Teachers, Burnout

**TRADUZA AGORA:**"""

        result = self._safe_generate(prompt, ', '.join(keywords))

        result = result.replace('\n', ', ')
        result = re.sub(r'[0-9]+\.\s*', '', result)
        translated = [t.strip().strip('"').strip("'") for t in result.split(',') if t.strip()]

        if len(translated) != len(keywords):
            return keywords

        return translated

    def create_search_string_with_objective(self, tema: str,
                                           original_keywords: List[str],
                                           suggested_keywords: str) -> Tuple[str, str]:
        """Cria string de busca otimizada com lógica booleana"""
        suggested_list = [s.strip() for s in suggested_keywords.split(',') if s.strip()]
        all_keywords = original_keywords + suggested_list

        prompt = f"""Você é especialista em recuperação de informação científica.

**CONTEXTO:**
Tema da pesquisa: {tema}
Termos disponíveis: {', '.join(all_keywords)}

**TAREFA:**
Crie uma string de busca em INGLÊS para bases científicas que:

1. **Selecione os melhores termos** (escolha 4-7 termos mais relevantes da lista)
2. **Use operadores booleanos:**
   - AND para termos obrigatórios
   - OR para sinônimos/alternativas (dentro de parênteses)
3. **Use aspas** para termos compostos (ex: "mental health")
4. **Agrupe** termos relacionados com parênteses
5. **Limite:** máximo 200 caracteres

**DEPOIS:**
Explique em 2-3 linhas o objetivo desta busca.

**FORMATO EXATO DA SAÍDA:**
STRING: (sua string aqui)
OBJETIVO: (explicação de 2-3 linhas)

**EXEMPLO:**
STRING: "teacher burnout" AND ("mental health" OR "psychological wellbeing") AND (school OR education)
OBJETIVO: Identificar estudos sobre esgotamento docente relacionados à saúde mental no contexto escolar, combinando descritores específicos do fenômeno com termos do ambiente educacional.

**AGORA CRIE PARA O TEMA '{tema}':**"""

        response = self._safe_generate(prompt, "")

        string_match = re.search(r'STRING:\s*(.+?)(?=OBJETIVO:|$)', response, re.DOTALL | re.IGNORECASE)
        obj_match = re.search(r'OBJETIVO:\s*(.+)', response, re.DOTALL | re.IGNORECASE)

        if string_match and obj_match:
            search_str = string_match.group(1).strip()
            search_str = search_str.replace('```', '').replace('\n', ' ')
            search_str = re.sub(r'\s+', ' ', search_str).strip()

            objective = obj_match.group(1).strip()
        else:
            main_terms = ' AND '.join([f'"{k}"' for k in original_keywords[:3]])
            sugg_terms = ' OR '.join([f'"{s}"' for s in suggested_list[:3]])

            if sugg_terms:
                search_str = f"{main_terms} AND ({sugg_terms})"
            else:
                search_str = main_terms

            objective = f"Identificar estudos que investigam {tema}, combinando descritores específicos do fenômeno com termos técnicos do contexto."

        return search_str, objective

    def create_glossary_and_interpretation(self, concepts: List[str],
                                          tema: str, primeiro_nome: str) -> Tuple[str, str]:
        """Cria glossário técnico e interpretação detalhada do grafo"""
        if not concepts or len(concepts) < 3:
            return ("Poucos conceitos identificados para análise detalhada.",
                    "Dados insuficientes para interpretação da rede conceitual.")

        concepts = concepts[:9]
        concepts_list = '\n'.join([f"{i+1}. {c}" for i, c in enumerate(concepts)])

        glossary_prompt = f"""Você é um especialista criando um glossário técnico.

**CONCEITOS IDENTIFICADOS NA REDE BIBLIOMÉTRICA:**
{concepts_list}

**TEMA DO PROJETO:** {tema}

---

**TAREFA:**
Para CADA um dos {len(concepts)} conceitos acima, crie uma entrada de glossário.

**FORMATO OBRIGATÓRIO PARA CADA ENTRADA:**

[Número]. **[Termo em Inglês]** (Tradução em Português) - [Definição técnica de 2-3 linhas]

**REGRAS:**
- Termo em inglês em **negrito**
- Tradução em português entre (parênteses) SEM negrito
- Traço " - " após os parênteses
- Definição clara, técnica e específica
- Relacionar com o tema '{tema}' quando possível
- Ordem alfabética pelo termo em INGLÊS
- NÃO pular nenhum conceito
- PROIBIDO usar frases clichê como "Com certeza", "Sem dúvida", "É claro que" ou similares
- Seja direto e técnico

**EXEMPLO DO FORMATO:**
1. **Anxiety** (Ansiedade) - Estado emocional caracterizado por preocupação excessiva, tensão e sintomas físicos de estresse. No contexto de {tema}, este conceito contribui para compreender as dimensões psicológicas do fenômeno investigado.

2. **Educational Psychology** (Psicologia Educacional) - Ramo da psicologia que investiga processos de ensino-aprendizagem, desenvolvimento cognitivo e fatores que influenciam o desempenho acadêmico. Permite análise multifacetada das questões relacionadas a {tema}.

**AGORA CRIE O GLOSSÁRIO COMPLETO PARA TODOS OS {len(concepts)} CONCEITOS:**"""

        interpretation_prompt = f"""Você é um cientometrista analisando uma rede conceitual.

**CONTEXTO:**
Tema da pesquisa: {tema}
Aluno: {primeiro_nome}

**9 CONCEITOS MAIS CENTRAIS NA REDE (Miller, 7±2):**
{concepts_list}

---

**TAREFA:**
Escreva uma interpretação detalhada da rede em 3-4 parágrafos (mínimo 12 linhas).

**ESTRUTURA:**

**Parágrafo 1 - Estrutura Geral (3-4 linhas):**
- Quais são os 3-4 conceitos MAIS centrais?
- O que essa centralidade revela sobre o campo?
- Como o conhecimento está organizado?

**Parágrafo 2 - Clusters e Relações (3-4 linhas):**
- Como os conceitos se agrupam?
- Há subdimensões claras no tema?
- Que conexões são mais interessantes?

**Parágrafo 3 - Implicações para {primeiro_nome} (4-6 linhas):**
- Como essa estrutura pode orientar o delineamento do escopo?
- Há lacunas que poderiam ser exploradas?
- Há oportunidades de pesquisa nas interseções?
- Recomendações específicas

**TOM:**
- Use "você" e "{primeiro_nome}"
- Cite conceitos específicos da rede (não seja genérico)
- Tom analítico mas acessível
- Oriente ações concretas
- NÃO use frases clichê como "Com certeza" ou similares

**COMECE COM:**
"{primeiro_nome}, o grafo de coocorrências revela a estrutura conceitual da literatura sobre {tema}..."

**ESCREVA AGORA A INTERPRETAÇÃO COMPLETA:**"""

        log_diagnostico("Gerando GLOSSÁRIO...", "info")
        glossary = self._safe_generate(
            glossary_prompt,
            self._generate_fallback_glossary(concepts, tema)
        )

        log_diagnostico("Gerando INTERPRETAÇÃO...", "info")
        interpretation = self._safe_generate(
            interpretation_prompt,
            self._generate_fallback_interpretation(concepts, tema, primeiro_nome)
        )

        return glossary, interpretation

    def _generate_fallback_glossary(self, concepts: List[str], tema: str) -> str:
        """Gera glossário fallback"""
        entries = []
        for i, concept in enumerate(concepts, 1):
            entries.append(f"{i}. **{concept}** - Conceito identificado na rede de coocorrências relacionado ao tema {tema}. Este termo aparece frequentemente na literatura científica sobre o assunto.")
        return "\n\n".join(entries)

    def _generate_fallback_interpretation(self, concepts: List[str], tema: str, primeiro_nome: str) -> str:
        """Gera interpretação fallback"""
        top_concepts = ', '.join(concepts[:4])
        return f"""{primeiro_nome}, o grafo de coocorrências revela a estrutura conceitual da literatura sobre {tema}, destacando {top_concepts} como conceitos centrais.

A presença desses termos indica que o campo de pesquisa está organizado em torno de dimensões teóricas e práticas bem estabelecidas. As conexões entre os conceitos sugerem interdisciplinaridade e múltiplas perspectivas de análise.

A análise desta rede pode orientar o delineamento do escopo da sua pesquisa, identificando áreas consolidadas onde há bastante literatura disponível, bem como possíveis lacunas nas interseções entre conceitos que podem representar oportunidades de investigação original."""


# ==================== ANALISADOR DE COOCORRÊNCIAS ====================
class CooccurrenceAnalyzer:
    """Analisador de redes"""

    def build_graph(self, concepts_lists: List[List[str]], min_cooc: int = 1) -> nx.Graph:
        G = nx.Graph()

        for concepts in concepts_lists:
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    if c1 != c2:
                        if G.has_edge(c1, c2):
                            G[c1][c2]['weight'] += 1
                        else:
                            G.add_edge(c1, c2, weight=1)

        weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_cooc]
        G.remove_edges_from(weak_edges)
        G.remove_nodes_from(list(nx.isolates(G)))

        return G

    def get_top_nodes(self, G: nx.Graph, n: int = 9) -> List[str]:
        if not G.nodes():
            return []

        centrality = nx.degree_centrality(G)
        return [node for node, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]]

    def visualize_graph(self, G: nx.Graph, top_n: int = 9, path: str = 'graph.png') -> str:
        top_nodes = self.get_top_nodes(G, top_n)
        if not top_nodes:
            return None

        Gs = G.subgraph(top_nodes).copy()
        pos = nx.spring_layout(Gs, k=0.5, iterations=50, seed=42)

        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(Gs))
            palette = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']

            color_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    color_map[node] = palette[i % len(palette)]

            colors = [color_map.get(n, '#3b82f6') for n in Gs.nodes()]
        except:
            colors = 'lightblue'

        centrality = nx.degree_centrality(Gs)
        sizes = [centrality[n] * 3000 + 300 for n in Gs.nodes()]

        plt.figure(figsize=(16, 12), facecolor='white')

        nx.draw_networkx_nodes(Gs, pos, node_size=sizes, node_color=colors,
                              alpha=0.85, edgecolors='white', linewidths=2.5)
        nx.draw_networkx_edges(Gs, pos, alpha=0.25, edge_color='gray')
        nx.draw_networkx_labels(Gs, pos, font_size=11, font_weight='bold',
                               font_family='sans-serif')

        plt.title("9 conceitos - (Miller, 7±2)", 
                 fontsize=20, fontweight='bold', pad=25)
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return path


# ==================== PIPELINE PRINCIPAL ====================
class ResearchScopePipeline:
    """Pipeline completo com diagnóstico"""

    def __init__(self, email: str):
        self.openalex = OpenAlexClient(email)
        self.gemini = GeminiQueryGenerator()
        self.analyzer = CooccurrenceAnalyzer()

    def process(self, nome: str, tema: str, questao: str, keywords: List[str]) -> Dict:
        """Executa pipeline com DIAGNÓSTICO VISUAL"""
        
        st.markdown("---")
        st.markdown("### 🔍 DIAGNÓSTICO DO PIPELINE")
        
        primeiro_nome = nome.split()[0] if nome else "estudante"

        # 1. Avaliação
        log_diagnostico("Etapa 1/7: Gerando avaliação do projeto...", "info")
        full_report = self.gemini.generate_full_report(nome, tema, questao, keywords)

        # 2. Termos complementares
        log_diagnostico("Etapa 2/7: Gerando termos complementares...", "info")
        suggested = self.gemini.generate_suggested_keywords(nome, tema, questao, keywords)

        # 3. String de busca
        log_diagnostico("Etapa 3/7: Criando string de busca...", "info")
        search_str, objetivo = self.gemini.create_search_string_with_objective(tema, keywords, suggested)

        # 4. Buscar artigos
        log_diagnostico("Etapa 4/7: Buscando artigos no OpenAlex...", "info")
        articles = self.openalex.search_articles(search_str, 500)
        log_diagnostico(f"Artigos encontrados: {len(articles)}", "success" if len(articles) > 0 else "warning")

        if len(articles) == 0:
            log_diagnostico("Tentando busca alternativa...", "warning")
            translated = self.gemini.translate_keywords_to_english(keywords)
            alt_search = ' AND '.join([f'"{t}"' for t in translated[:3]])
            articles = self.openalex.search_articles(alt_search, 500)

        # 5. Extrair conceitos
        log_diagnostico("Etapa 5/7: Extraindo conceitos...", "info")
        concepts_lists = self.openalex.extract_concepts_for_cooccurrence(articles)
        log_diagnostico(f"Conceitos extraídos de {len(concepts_lists)} artigos", "success")

        # 6. Construir grafo
        log_diagnostico("Etapa 6/7: Construindo grafo...", "info")
        G = self.analyzer.build_graph(concepts_lists, min_cooc=1)
        log_diagnostico(f"Grafo: {len(G.nodes())} nós, {len(G.edges())} arestas", "success")

        # 7. Visualizar e interpretar
        log_diagnostico("Etapa 7/7: Gerando visualização e análise...", "info")
        viz_path = self.analyzer.visualize_graph(G, 9)
        top_concepts = self.analyzer.get_top_nodes(G, 9)
        log_diagnostico(f"Top conceitos: {top_concepts[:5]}...", "info")

        glossary, interpretation = self.gemini.create_glossary_and_interpretation(
            top_concepts, tema, primeiro_nome
        )

        log_diagnostico("PIPELINE CONCLUÍDO!", "success")
        st.markdown("---")

        return {
            'full_report': full_report,
            'search_string': search_str,
            'search_objective': objetivo,
            'articles_count': len(articles),
            'graph_stats': {'nodes': len(G.nodes()), 'edges': len(G.edges())},
            'visualization_path': viz_path,
            'glossary': glossary,
            'graph_interpretation': interpretation,
            'top_concepts': top_concepts,
            'raw_articles': articles,
            'concepts_lists': concepts_lists,
            'graph': G
        }


# Variável global
OPENALEX_EMAIL = st.secrets.get("OPENALEX_EMAIL", "")