# -*- coding: utf-8 -*-

import google.generativeai as genai
import requests
import json
import re
import time
import traceback
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== CONFIGURAÇÃO ====================
import streamlit as st

# Ler credenciais do arquivo secrets.toml
GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENALEX_EMAIL = st.secrets["OPENALEX_EMAIL"]

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# ==================== CLIENTE OPENALEX ====================
class OpenAlexClient:
    """Cliente para buscar artigos científicos no OpenAlex"""

    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org/works"
        self.email = email

    def normalize_query(self, query: str) -> str:
        """Normaliza a query de busca"""
        query = re.sub(r'"+', '"', query)
        query = re.sub(r'\s*(AND|OR|NOT)\s*', r' \1 ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+', ' ', query).strip()
        return query

    def search_articles(self, query: str, limit: int = 500) -> List[Dict]:
        query = self.normalize_query(query)
        results = []
        print("Buscando artigos...")

        for page in range(1, 4):
            try:
                params = {
                    'search': query,
                    'per-page': 200,
                    'page': page,
                    'mailto': self.email,
                    'filter': 'type:article'
                }

                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                page_results = data.get('results', [])

                print(f"Pagina {page}: {len(page_results)} artigos")

                for work in page_results:
                    doi = work.get('doi', '')
                    url = doi if doi else work.get('id', '')

                    results.append({
                        'title': work.get('title', ''),
                        'year': work.get('publication_year'),
                        'doi': doi,
                        'url': url,
                        'concepts': [
                            {
                                'name': c['display_name'],
                                'score': c['score'],
                                'level': c['level']
                            }
                            for c in work.get('concepts', [])
                            if c.get('score', 0) > 0.3
                        ]
                    })

                if len(results) >= limit:
                    break

            except Exception as e:
                print(f"Erro na pagina {page}: {str(e)}")
                break

        print(f"Total: {len(results)} artigos")
        return results[:limit]

    def extract_concepts_for_cooccurrence(self, articles: List[Dict],
                                         min_score: float = 0.35,
                                         min_level: int = 0) -> List[List[str]]:
        """Extrai conceitos dos artigos"""
        concepts_lists = []

        for article in articles:
            concepts = [
                c['name'] for c in article.get('concepts', [])
                if c['score'] >= min_score and c['level'] >= min_level
            ]
            if concepts:
                concepts_lists.append(concepts)

        print(f"  📊 {len(concepts_lists)} artigos com conceitos")
        return concepts_lists


# ==================== GERADOR COM GEMINI ====================
class GeminiQueryGenerator:
    """
    Gerador de análises usando Gemini AI.
    VERSÃO CORRIGIDA: Modelo estável + método de tradução
    """

    def __init__(self):
        try:
            self.model = genai.GenerativeModel(
                'gemini-2.5-pro',  # MODELO ESTÁVEL
                generation_config={
                    'temperature': 0.95,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 8192,
                }
            )
            # FIX: Changed to string concatenation to avoid outer f-string evaluation of self.model
            print("  ✅ Modelo Gemini inicializado: " + self.model.model_name + " (modo criativo)")
        except Exception as e:
            print(f"  ⚠️ Erro ao inicializar Gemini: {e}")
            self.model = None

    def _safe_generate(self, prompt: str, fallback: str = "", max_retries: int = 3) -> str:
        """Geração segura com retry e validação flexível"""
        if not self.model:
            print("  ⚠️ Modelo não disponível, usando fallback")
            return fallback

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)

                extracted_text = None

                if hasattr(response, 'text'):
                    try:
                        extracted_text = response.text
                    except:
                        pass

                if not extracted_text and hasattr(response, 'candidates'):
                    try:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content'):
                            parts = candidate.content.parts
                            if parts and len(parts) > 0:
                                extracted_text = parts[0].text
                    except:
                        pass

                if not extracted_text:
                    extracted_text = str(response)

                if extracted_text:
                    extracted_text = extracted_text.strip()
                    if len(extracted_text) >= 30 and extracted_text != "None":
                        return extracted_text

                if attempt < max_retries - 1:
                    print(f"  ⏳ Tentativa {attempt + 2}/{max_retries}...")
                    time.sleep(2)

            except Exception as e:
                print(f"  ⚠️ Tentativa {attempt + 1} falhou: {str(e)[:50]}")
                if attempt < max_retries - 1:
                    time.sleep(3)

        print(f"  ⚠️ Usando fallback após {max_retries} tentativas")
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

Escreva um parágrafo conversando com {primeiro_nome} sobre as palavras-chave que ele escolheu.

**DIRETRIZES (use seu julgamento profissional):**

• Comece com: "{primeiro_nome}, as palavras-chave que você designou para o projeto..."

• **Seja autêntico e direto:**
  - Se o projeto fizer sentido, comente o que está bom
  - Se houver problemas evidentes (tema absurdo, questão impossível, palavras sem relação),
    aponte isso com clareza mas cuidado
  - Se palavras forem muito amplas (ex: "Escola", "Psicologia"), diga quais e por quê
  - Se houver redundância entre termos, mostre
  - Se faltar algo importante, sugira especificamente o quê

• **Tom de conversa:**
  - Use "você" e o primeiro nome
  - Honesto mas respeitoso
  - Como um professor que realmente se importa com o aluno
  - NÃO use linguagem de parecer formal ou formulário padrão
  - Pode usar frases como "vejo que...", "considere...", "seria interessante..."

• **Seja específico:**
  - Comente sobre ESTAS palavras-chave específicas
  - Não use frases genéricas que servem para qualquer projeto
  - Se um termo for bom, diga por quê
  - Se um termo for problemático, explique o problema

• **Encerre com:**
  "Recomendo que você observe atentamente o grafo de coocorrências apresentado adiante,
   pois ele pode revelar relações importantes entre conceitos que ajudarão a refinar suas
   palavras-chave e a delimitar melhor o escopo da sua pesquisa."

**IMPORTANTE:**
- Projetos com temas claramente absurdos ou questões impossíveis merecem feedback honesto
- Não finja que algo inviável é viável
- Seja gentil mas não desonesto

---

Escreva agora o parágrafo para {primeiro_nome}:"""

        fallback = f"""{primeiro_nome}, as palavras-chave que você designou para o projeto ({keywords_str}) cobrem alguns aspectos do tema '{tema}'. Seria importante conversar com seu orientador para avaliar se esses termos capturam as nuances específicas da sua questão de pesquisa e se há necessidade de termos mais específicos ou complementares. Recomendo que você observe atentamente o grafo de coocorrências apresentado adiante, pois ele pode revelar relações importantes entre conceitos que ajudarão a refinar suas palavras-chave e a delimitar melhor o escopo da sua pesquisa."""

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
        """Traduz palavras-chave do português para inglês.
        Método necessário para compatibilidade com pipeline.
        """
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

        # Limpar e separar
        result = result.replace('\n', ', ')
        result = re.sub(r'[0-9]+\.\s*', '', result)  # Remove numeração
        translated = [t.strip().strip('"').strip("' ") for t in result.split(',') if t.strip()]

        # Se não conseguiu traduzir ou número diferente, retornar original
        if len(translated) != len(keywords):
            print(f"  ⚠️ Tradução inconsistente, usando termos originais")
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

        concepts = concepts[:15]
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

**EXEMPLO DO FORMATO:**
1. **Anxiety** (Ansiedade) - Estado emocional caracterizado por preocupação excessiva, tensão e sintomas físicos de estresse. No contexto de {tema}, este conceito contribui para compreender as dimensões psicológicas do fenômeno investigado.

2. **Educational Psychology** (Psicologia Educacional) - Ramo da psicologia que investiga processos de ensino-aprendizagem, desenvolvimento cognitivo e fatores que influenciam o desempenho acadêmico. Permite análise multifacetada das questões relacionadas a {tema}.

**AGORA CRIE O GLOSSÁRIO COMPLETO PARA TODOS OS {len(concepts)} CONCEITOS:**"""

        interpretation_prompt = f"""Você é um cientometrista analisando uma rede conceitual.

**CONTEXTO:**
Tema da pesquisa: {tema}
Aluno: {primeiro_nome}

**15 CONCEITOS MAIS CENTRAIS NA REDE:**
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
- Há oportunidades de pesquisa nas intersecções?
- Recomendações específicas

**TOM:**
- Use "você" e "{primeiro_nome}"
- Cite conceitos específicos da rede (não seja genérico)
- Tom analítico mas acessível
- Oriente ações concretas

**COMECE COM:**
"{primeiro_nome}, o grafo de coocorrências revela a estrutura conceitual da literatura sobre {tema}..."

**ESCREVA AGORA A INTERPRETAÇÃO COMPLETA:**"""

        print("  🔤 Gerando glossário...")
        glossary = self._safe_generate(
            glossary_prompt,
            f"Glossário técnico dos {len(concepts)} conceitos centrais identificados na rede de coocorrências."
        )

        print("  📊 Gerando interpretação da rede...")
        interpretation = self._safe_generate(
            interpretation_prompt,
            f"{primeiro_nome}, o grafo de coocorrências revela a estrutura conceitual da literatura sobre {tema}, destacando {', '.join([c for c in concepts[:4]])} como conceitos centrais. A análise desta rede pode orientar o delineamento do escopo da sua pesquisa, identificando áreas consolidadas e possíveis lacunas para investigação."
        )

        return glossary, interpretation


# ==================== ANALISADOR DE COOCORRÊNCIAS ====================
class CooccurrenceAnalyzer:
    """Analisador de redes"""

    def build_graph(self, concepts_lists: List[List[str]], min_cooc: int = 1) -> nx.Graph:
        """Constrói grafo"""
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

        print(f"  🕸️ Grafo: {len(G.nodes())} nós, {len(G.edges())} arestas")
        return G

    def get_top_nodes(self, G: nx.Graph, n: int = 15) -> List[str]:
        """Nós mais centrais"""
        if not G.nodes():
            return []

        centrality = nx.degree_centrality(G)
        return [node for node, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]]

    def visualize_graph(self, G: nx.Graph, top_n: int = 15, path: str = 'graph.png') -> str:
        """Visualização"""
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

        plt.title("Rede de Coocorrência de Conceitos", fontsize=20,
                 fontweight='bold', pad=25)
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  🎨 Visualização: {path}")
        return path


# ==================== PIPELINE PRINCIPAL ====================
class ResearchScopePipeline:
    """Pipeline completo"""

    def __init__(self, email: str):
        self.openalex = OpenAlexClient(email)
        self.gemini = GeminiQueryGenerator()
        self.analyzer = CooccurrenceAnalyzer()

    def process(self, nome: str, tema: str, questao: str, keywords: List[str]) -> Dict:
        """Executa pipeline completo"""
        print("\n" + "="*80)
        print("🚀 PIPELINE FINAL - VERSÃO DEFINITIVA")
        print("="*80 + "\n")

        primeiro_nome = nome.split()[0] if nome else "estudante"

        # 1. Avaliação com 2 parágrafos
        print("📝 Etapa 1/7: Avaliação completa (tema+questão+palavras)...")
        full_report = self.gemini.generate_full_report(nome, tema, questao, keywords)

        # 2. Termos complementares
        print("💡 Etapa 2/7: Gerando termos complementares...")
        suggested = self.gemini.generate_suggested_keywords(nome, tema, questao, keywords)
        print(f"     → Sugeridos: {suggested[:60]}...")

        # 3. String 100% inglês
        print("🔎 Etapa 3/7: Criando string 100% em inglês...")
        search_str, objetivo = self.gemini.create_search_string_with_objective(tema, keywords, suggested)

        # 4. Buscar artigos
        print("📚 Etapa 4/7: Buscando artigos no OpenAlex...")
        articles = self.openalex.search_articles(search_str, 500)

        if len(articles) == 0:
            print("  ⚠️ Sem resultados. Tentando com termos traduzidos...")
            translated = self.gemini.translate_keywords_to_english(keywords)
            alt_search = ' AND '.join([f'"{t}"' for t in translated[:3]])
            articles = self.openalex.search_articles(alt_search, 500)

        # 5. Extrair conceitos
        print("🔬 Etapa 5/7: Extraindo conceitos...")
        concepts_lists = self.openalex.extract_concepts_for_cooccurrence(articles)

        # 6. Construir grafo
        print("🕸️ Etapa 6/7: Construindo grafo...")
        G = self.analyzer.build_graph(concepts_lists, min_cooc=1)

        # 7. Visualizar e interpretar
        print("🎨 Etapa 7/7: Gerando visualização e glossário...")
        viz_path = self.analyzer.visualize_graph(G, 15)
        top_concepts = self.analyzer.get_top_nodes(G, 15)

        glossary, interpretation = self.gemini.create_glossary_and_interpretation(
            top_concepts, tema, primeiro_nome
        )

        print("\n" + "="*80)
        print("✅ PIPELINE CONCLUÍDO!")
        print("="*80 + "\n")

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
