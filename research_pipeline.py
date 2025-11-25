# -*- coding: utf-8 -*-
"""
RESEARCH PIPELINE - VERSÃO CORRIGIDA
=====================================
Correções aplicadas:
1. Logging detalhado para diagnóstico
2. Tratamento de erros melhorado no Gemini
3. Prompts otimizados (mais concisos)
4. Verificação de bloqueio de segurança
5. Fallbacks mais informativos
"""

import time
import re
from typing import List, Dict, Tuple
import requests
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# ==================== CLIENTE OPENALEX ====================
class OpenAlexClient:
    """Cliente para buscar artigos no OpenAlex"""

    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://api.openalex.org/works"

    def normalize_query(self, query: str) -> str:
        """Normaliza query mantendo operadores booleanos"""
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    def search_articles(self, query: str, limit: int = 500) -> List[Dict]:
        """Busca artigos na API do OpenAlex"""
        results = []
        
        print(f"  🔍 Buscando: {query[:100]}...")
        
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
                    print(f"  ⚠️ Erro HTTP {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"  ⚠️ Erro: {str(e)[:50]}")
                break
        
        print(f"  ✅ {len(results)} artigos encontrados")
        return results[:limit]

    def extract_concepts_for_cooccurrence(self, articles: List[Dict],
                                         min_score: float = 0.35,
                                         min_level: int = 0) -> List[List[str]]:
        """Extrai conceitos dos artigos para análise de coocorrência"""
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


# ==================== GERADOR GEMINI - VERSÃO CORRIGIDA ====================
class GeminiQueryGenerator:
    """
    Gerador de análises usando Gemini AI.
    VERSÃO CORRIGIDA COM LOGGING E TRATAMENTO DE ERROS MELHORADO
    """

    def __init__(self):
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY", "")
            
            # LOG: Verificar se a chave foi encontrada
            if api_key:
                print(f"  ✅ GEMINI_API_KEY encontrada (tamanho: {len(api_key)} chars)")
            else:
                print("  ❌ GEMINI_API_KEY NÃO encontrada nos secrets!")
                
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(
                'gemini-2.5-pro',  
                generation_config={
                    'temperature': 1.2, 
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 8192, 
                }
            )
            print(f"  ✅ Modelo Gemini inicializado: {self.model.model_name}")
        except Exception as e:
            print(f"  ❌ Erro ao inicializar Gemini: {e}")
            self.model = None

    def _safe_generate(self, prompt: str, fallback: str = "", max_retries: int = 3) -> str:
        """
        Geração segura com retry, logging detalhado e tratamento de bloqueios
        """
        if not self.model:
            print("  ❌ Modelo não disponível, usando fallback")
            return fallback

        for attempt in range(max_retries):
            try:
                print(f"  🔄 Tentativa {attempt + 1}/{max_retries}...")
                print(f"     Tamanho do prompt: {len(prompt)} chars")
                
                response = self.model.generate_content(prompt)
                
                # LOG: Verificar resposta
                print(f"     Resposta recebida: {type(response)}")
                
                # Verificar bloqueio de segurança
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    print(f"     Prompt feedback: {feedback}")
                    if hasattr(feedback, 'block_reason') and feedback.block_reason:
                        print(f"  ⚠️ Prompt BLOQUEADO: {feedback.block_reason}")
                        continue

                extracted_text = None

                # Método 1: Acesso direto via .text
                if hasattr(response, 'text'):
                    try:
                        extracted_text = response.text
                        print(f"     Método .text: {len(extracted_text) if extracted_text else 0} chars")
                    except ValueError as ve:
                        print(f"     Método .text falhou: {ve}")

                # Método 2: Via candidates
                if not extracted_text and hasattr(response, 'candidates') and response.candidates:
                    try:
                        candidate = response.candidates[0]
                        
                        # Verificar finish_reason
                        if hasattr(candidate, 'finish_reason'):
                            print(f"     Finish reason: {candidate.finish_reason}")
                        
                        # Verificar safety_ratings
                        if hasattr(candidate, 'safety_ratings'):
                            for rating in candidate.safety_ratings:
                                if hasattr(rating, 'probability') and str(rating.probability) in ['HIGH', 'MEDIUM']:
                                    print(f"     ⚠️ Safety rating: {rating.category} = {rating.probability}")
                        
                        if hasattr(candidate, 'content') and candidate.content:
                            parts = candidate.content.parts
                            if parts and len(parts) > 0:
                                extracted_text = parts[0].text
                                print(f"     Método candidates: {len(extracted_text) if extracted_text else 0} chars")
                    except Exception as ce:
                        print(f"     Método candidates falhou: {ce}")

                # Método 3: Conversão string (último recurso)
                if not extracted_text:
                    extracted_text = str(response)
                    print(f"     Método str(): {len(extracted_text)} chars")

                # Validação final
                if extracted_text:
                    extracted_text = extracted_text.strip()
                    
                    # Remover possíveis artefatos
                    if extracted_text.startswith("response:"):
                        extracted_text = ""
                    
                    if len(extracted_text) >= 50:  # Aumentado de 30 para 50
                        print(f"  ✅ Texto válido extraído: {len(extracted_text)} chars")
                        print(f"     Preview: {extracted_text[:100]}...")
                        return extracted_text
                    else:
                        print(f"  ⚠️ Texto muito curto: {len(extracted_text)} chars")
                        print(f"     Conteúdo: {extracted_text[:200]}")

                if attempt < max_retries - 1:
                    print(f"  ⏳ Aguardando 3s antes de retry...")
                    time.sleep(3)

            except Exception as e:
                print(f"  ❌ Exceção na tentativa {attempt + 1}: {type(e).__name__}: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(3)

        print(f"  ⚠️ USANDO FALLBACK após {max_retries} tentativas")
        return fallback

    def generate_full_report(self, nome: str, tema: str, questao: str,
                            keywords: List[str]) -> str:
        """
        Gera avaliação crítica e construtiva do projeto
        PROMPT OTIMIZADO - mais conciso
        """
        keywords_str = ', '.join(keywords)
        primeiro_nome = nome.split()[0] if nome else "estudante"

        # PROMPT MAIS CONCISO
        prompt = f"""Você é um professor orientando {primeiro_nome} em seu projeto de pesquisa.

PROJETO:
- Tema: {tema}
- Questão: {questao}
- Palavras-chave: {keywords_str}

TAREFA: Escreva 2 parágrafos de feedback.

PARÁGRAFO 1: Comece com "{primeiro_nome}, as palavras-chave que você designou para o projeto..." 
Analise especificamente as palavras-chave. Seja honesto sobre pontos fortes e fracos.

PARÁGRAFO 2: Comente sobre a questão de pesquisa. Analise clareza e viabilidade.
Termine com: "Recomendo que você converse com seu orientador sobre esses pontos e observe atentamente o grafo de coocorrências apresentado adiante, pois ele pode revelar relações importantes entre conceitos que ajudarão a refinar suas palavras-chave e a delimitar melhor o escopo da sua pesquisa."

Use tom conversacional. Seja específico. Evite frases genéricas."""

        fallback = f"""{primeiro_nome}, as palavras-chave que você designou para o projeto ({keywords_str}) cobrem alguns aspectos do tema '{tema}'. No entanto, seria importante avaliar se esses termos capturam as nuances específicas da sua questão de pesquisa e se há necessidade de termos mais específicos ou complementares.

Sobre sua questão de pesquisa, '{questao}', é fundamental verificar se está suficientemente delimitada e se oferece um caminho claro para investigação. Recomendo que você converse com seu orientador sobre esses pontos e observe atentamente o grafo de coocorrências apresentado adiante, pois ele pode revelar relações importantes entre conceitos que ajudarão a refinar suas palavras-chave e a delimitar melhor o escopo da sua pesquisa."""

        return self._safe_generate(prompt, fallback)

    def generate_suggested_keywords(self, nome: str, tema: str, questao: str,
                                   keywords: List[str]) -> str:
        """Sugere palavras-chave complementares em inglês técnico"""
        keywords_str = ', '.join(keywords)

        prompt = f"""Sugira 4-6 termos técnicos EM INGLÊS complementares para esta pesquisa:

Tema: {tema}
Palavras atuais: {keywords_str}

Retorne APENAS os termos separados por vírgula, sem explicações.
Exemplo: cognitive load, metacognition, learning strategies"""

        fallback = "research methods, empirical studies, theoretical framework, scientific literature"

        result = self._safe_generate(prompt, fallback)
        result = result.replace('\n', ', ').strip()
        return result

    def translate_keywords_to_english(self, keywords: List[str]) -> List[str]:
        """Traduz palavras-chave do português para inglês."""
        keywords_str = ', '.join(keywords)

        prompt = f"""Traduza para inglês acadêmico: {keywords_str}

Retorne APENAS os termos traduzidos, separados por vírgula, na mesma ordem."""

        result = self._safe_generate(prompt, ', '.join(keywords))
        result = result.replace('\n', ', ')
        result = re.sub(r'[0-9]+\.\s*', '', result)
        translated = [t.strip().strip('"').strip("'") for t in result.split(',') if t.strip()]

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

        prompt = f"""Crie uma string de busca científica para o tema: {tema}

Termos disponíveis: {', '.join(all_keywords)}

FORMATO DA RESPOSTA:
STRING: (sua string com AND, OR, aspas para termos compostos)
OBJETIVO: (2-3 linhas explicando o objetivo da busca)

Exemplo:
STRING: "teacher burnout" AND ("mental health" OR wellbeing) AND school
OBJETIVO: Identificar estudos sobre esgotamento docente no contexto escolar."""

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
            search_str = f"{main_terms} AND ({sugg_terms})" if sugg_terms else main_terms
            objective = f"Identificar estudos que investigam {tema}, combinando descritores específicos do fenômeno com termos técnicos do contexto."

        return search_str, objective

    def create_glossary_and_interpretation(self, concepts: List[str],
                                          tema: str, primeiro_nome: str) -> Tuple[str, str]:
        """
        Cria glossário técnico e interpretação detalhada do grafo
        PROMPTS OTIMIZADOS - mais curtos e diretos
        """
        if not concepts or len(concepts) < 3:
            return ("Poucos conceitos identificados para análise detalhada.",
                    "Dados insuficientes para interpretação da rede conceitual.")

        concepts = concepts[:9]
        concepts_str = ', '.join(concepts)

        # ========== GLOSSÁRIO - PROMPT SIMPLIFICADO ==========
        glossary_prompt = f"""Crie um glossário técnico para estes conceitos do tema "{tema}":

{concepts_str}

Para CADA conceito, use este formato:
[Número]. **[Termo em Inglês]** (Tradução) - Definição técnica de 2-3 linhas relacionada ao tema.

Exemplo:
1. **Psychology** (Psicologia) - Ciência que estuda o comportamento e processos mentais. No contexto de {tema}, permite compreender aspectos cognitivos e emocionais envolvidos.

Crie entradas para TODOS os {len(concepts)} conceitos listados."""

        # ========== INTERPRETAÇÃO - PROMPT SIMPLIFICADO ==========
        interpretation_prompt = f"""{primeiro_nome}, analise esta rede de conceitos sobre "{tema}".

Conceitos centrais: {concepts_str}

Escreva 3 parágrafos:

1. ESTRUTURA: Quais são os 3-4 conceitos mais centrais? O que isso revela sobre o campo?

2. RELAÇÕES: Como os conceitos se agrupam? Que conexões são interessantes?

3. RECOMENDAÇÕES: Como essa estrutura pode orientar o delineamento do projeto de {primeiro_nome}? Que lacunas podem ser exploradas?

Use tom conversacional e cite conceitos específicos da lista."""

        print("  📖 Gerando glossário...")
        glossary = self._safe_generate(
            glossary_prompt,
            self._generate_fallback_glossary(concepts, tema)
        )

        print("  📊 Gerando interpretação...")
        interpretation = self._safe_generate(
            interpretation_prompt,
            self._generate_fallback_interpretation(concepts, tema, primeiro_nome)
        )

        return glossary, interpretation

    def _generate_fallback_glossary(self, concepts: List[str], tema: str) -> str:
        """Gera um glossário fallback mais informativo"""
        entries = []
        for i, concept in enumerate(concepts, 1):
            entries.append(f"{i}. **{concept}** - Conceito identificado na rede de coocorrências relacionado ao tema {tema}. Este termo aparece frequentemente na literatura científica sobre o assunto.")
        return "\n\n".join(entries)

    def _generate_fallback_interpretation(self, concepts: List[str], tema: str, primeiro_nome: str) -> str:
        """Gera uma interpretação fallback mais informativa"""
        top_concepts = ', '.join(concepts[:4])
        return f"""{primeiro_nome}, o grafo de coocorrências revela a estrutura conceitual da literatura sobre {tema}, destacando {top_concepts} como conceitos centrais.

A presença desses termos indica que o campo de pesquisa está organizado em torno de dimensões teóricas e práticas bem estabelecidas. As conexões entre os conceitos sugerem interdisciplinaridade e múltiplas perspectivas de análise.

A análise desta rede pode orientar o delineamento do escopo da sua pesquisa, identificando áreas consolidadas onde há bastante literatura disponível, bem como possíveis lacunas nas interseções entre conceitos que podem representar oportunidades de investigação original."""


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

    def get_top_nodes(self, G: nx.Graph, n: int = 9) -> List[str]:
        """Nós mais centrais - Default de 9 termos (Miller, 7±2)"""
        if not G.nodes():
            return []

        centrality = nx.degree_centrality(G)
        return [node for node, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]]

    def visualize_graph(self, G: nx.Graph, top_n: int = 9, path: str = 'graph.png') -> str:
        """Visualização - Default de 9 termos (Miller, 7±2)"""
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
        print("🚀 PIPELINE DELINÉIA - VERSÃO CORRIGIDA COM LOGGING")
        print("="*80 + "\n")

        primeiro_nome = nome.split()[0] if nome else "estudante"

        # 1. Avaliação
        print("📝 Etapa 1/7: Avaliação completa...")
        full_report = self.gemini.generate_full_report(nome, tema, questao, keywords)

        # 2. Termos complementares
        print("\n💡 Etapa 2/7: Gerando termos complementares...")
        suggested = self.gemini.generate_suggested_keywords(nome, tema, questao, keywords)
        print(f"     → Sugeridos: {suggested[:60]}...")

        # 3. String de busca
        print("\n🔎 Etapa 3/7: Criando string de busca...")
        search_str, objetivo = self.gemini.create_search_string_with_objective(tema, keywords, suggested)

        # 4. Buscar artigos
        print("\n📚 Etapa 4/7: Buscando artigos no OpenAlex...")
        articles = self.openalex.search_articles(search_str, 500)

        if len(articles) == 0:
            print("  ⚠️ Sem resultados. Tentando com termos traduzidos...")
            translated = self.gemini.translate_keywords_to_english(keywords)
            alt_search = ' AND '.join([f'"{t}"' for t in translated[:3]])
            articles = self.openalex.search_articles(alt_search, 500)

        # 5. Extrair conceitos
        print("\n🔬 Etapa 5/7: Extraindo conceitos...")
        concepts_lists = self.openalex.extract_concepts_for_cooccurrence(articles)

        # 6. Construir grafo
        print("\n🕸️ Etapa 6/7: Construindo grafo...")
        G = self.analyzer.build_graph(concepts_lists, min_cooc=1)

        # 7. Visualizar e interpretar
        print("\n🎨 Etapa 7/7: Gerando visualização e análise...")
        viz_path = self.analyzer.visualize_graph(G, 9)
        top_concepts = self.analyzer.get_top_nodes(G, 9)
        print(f"     Top conceitos: {top_concepts}")

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


# Variável global necessária
import streamlit as st
OPENALEX_EMAIL = st.secrets.get("OPENALEX_EMAIL", "")