import pandas as pd
import io
import json
import networkx as nx
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import streamlit as st

def _extract_authors(article):
    """Auxiliar: Extrai string limpa de autores."""
    try:
        authorships = article.get('authorships') or [] # Proteção contra None
        names = []
        for a in authorships:
            author_obj = a.get('author') or {}
            name = author_obj.get('display_name')
            if name: names.append(name)
        
        return ", ".join(names)
    except:
        return ""

def _extract_concepts_string(article):
    """Auxiliar: Extrai string limpa de conceitos/keywords."""
    try:
        concepts = article.get('concepts') or [] # Proteção contra None
        
        # Pega top 5 conceitos usando display_name ou name
        names = [c.get('display_name', c.get('name', '')) for c in concepts[:5]]
        return ", ".join([n for n in names if n])
    except:
        return ""

def _safe_get_source(article):
    """Auxiliar: Extrai nome da revista/fonte com segurança máxima."""
    try:
        loc = article.get('primary_location') or {}
        source = loc.get('source') or {}
        return source.get('display_name', '')
    except:
        return ""

def _get_year(article):
    """Auxiliar: Tenta pegar o ano de várias formas."""
    # Tenta o campo padrão
    year = article.get('publication_year')
    if year: return int(year)
    
    # Tenta extrair da data completa (YYYY-MM-DD)
    date = article.get('publication_date')
    if date and len(str(date)) >= 4:
        return str(date)[:4]
        
    return ""

def generate_excel(articles):
    """Gera Excel com metadados RICOS e TRATADOS."""
    flattened_data = []
    
    for art in articles:
        # Blindagem do DOI (transforma None em string vazia antes do replace)
        doi = (art.get('doi') or '').replace('https://doi.org/', '')
        
        # Garante título
        titulo = art.get('title') or art.get('display_name') or 'Sem título'

        flattened_data.append({
            'Título': titulo,
            'Ano': _get_year(art), # USANDO NOVA FUNÇÃO DE ANO
            'Autores': _extract_authors(art),
            'Revista/Fonte': _safe_get_source(art),
            'Citações': art.get('cited_by_count', 0),
            'DOI': doi,
            'Conceitos (Keywords)': _extract_concepts_string(art),
            'Tipo': art.get('type', ''),
            'Link': art.get('doi') or art.get('id', '')
        })
    
    df = pd.DataFrame(flattened_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
        worksheet = writer.sheets['Resultados']
        for idx, col in enumerate(df.columns):
            worksheet.set_column(idx, idx, 20)
            
    return output.getvalue()

def generate_bibtex(articles):
    """Gera BibTeX robusto."""
    db = BibDatabase()
    entries = []
    
    for i, art in enumerate(articles):
        authors = _extract_authors(art).split(', ')
        first_author = authors[0].split(' ')[-1] if authors and authors[0] else "Unknown"
        year = str(_get_year(art) or 'nd')
        
        clean_author = "".join(filter(str.isalnum, first_author))
        citation_key = f"{clean_author}{year}{i}"
        
        # Blindagem do DOI
        doi = (art.get('doi') or '').replace('https://doi.org/', '')
        
        entry = {
            'ENTRYTYPE': 'article',
            'ID': citation_key,
            'title': art.get('title', 'Sem título'),
            'year': year,
            'author': " and ".join(authors),
            'journal': _safe_get_source(art),
            'doi': doi,
            'url': art.get('doi') or art.get('id', ''),
            'keywords': _extract_concepts_string(art)
        }
        
        entry = {k: v for k, v in entry.items() if v}
        entries.append(entry)
        
    db.entries = entries
    writer = BibTexWriter()
    return writer.write(db)

def generate_ris(articles):
    """Gera arquivo RIS."""
    lines = []
    for art in articles:
        lines.append("TY  - JOUR")
        lines.append(f"TI  - {art.get('title', '')}")
        
        raw_authors = _extract_authors(art).split(', ')
        for auth in raw_authors:
            if auth: lines.append(f"AU  - {auth}")
            
        lines.append(f"PY  - {_get_year(art)}///")
        lines.append(f"JO  - {_safe_get_source(art)}")
        
        # Blindagem do DOI
        doi = (art.get('doi') or '').replace('https://doi.org/', '')
        if doi: lines.append(f"DO  - {doi}")
        
        kws = _extract_concepts_string(art).split(', ')
        for kw in kws:
            if kw: lines.append(f"KW  - {kw.strip()}")
            
        lines.append(f"UR  - {art.get('doi') or art.get('id', '')}")
        lines.append("ER  - \n")
        
    return "\n".join(lines)

def generate_pajek_net(graph):
    """Gera arquivo .net para Pajek/VOSviewer."""
    if not graph: return b""
    
    if graph.is_directed():
        G = graph.to_undirected()
    else:
        G = graph
        
    output = io.BytesIO()
    try:
        lines = [f"*Vertices {G.number_of_nodes()}"]
        nodes = list(G.nodes())
        node_map = {name: i+1 for i, name in enumerate(nodes)}
        
        for name in nodes:
            safe_name = name.replace('"', "'")
            lines.append(f'{node_map[name]} "{safe_name}"')
            
        lines.append("*Edges")
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            lines.append(f"{node_map[u]} {node_map[v]} {weight}")
            
        output.write("\n".join(lines).encode('utf-8'))
        return output.getvalue()
    except Exception as e:
        return f"Erro: {str(e)}".encode('utf-8')

def listar_grafos_salvos(sheet_obj, id_usuario_filtro=None):
    """
    Lista grafos. Se id_usuario_filtro for passado, retorna APENAS os desse usuário.
    """
    try:
        worksheets = sheet_obj.worksheets()
        grafos = []
        
        # Prepara o ID curto para comparação (mesma lógica do salvamento)
        id_busca = None
        if id_usuario_filtro:
            # Pega os últimos 8 caracteres se for longo, ou o próprio ID
            id_busca = id_usuario_filtro[-8:] if len(id_usuario_filtro) > 8 else id_usuario_filtro

        for ws in worksheets:
            if ws.title.startswith("G_"):
                # Se tiver filtro, verifica se o ID CURTO está no título da aba
                if id_busca:
                    if id_busca in ws.title:
                        grafos.append({"title": ws.title, "obj": ws})
                else:
                    # Sem filtro (admin/debug), não mostra nada por segurança
                    pass 
                    
        return sorted(grafos, key=lambda x: x['title'], reverse=True)
    except Exception as e:
        print(f"Erro ao listar: {e}")
        return []

def carregar_grafo_do_sheets(worksheet):
    """
    Lê uma aba do Sheets, ignora os metadados do cabeçalho e retorna um DataFrame das arestas.
    """
    try:
        # Lê todos os valores como uma matriz (lista de listas)
        all_values = worksheet.get_all_values()
        
        # Procura onde começam os dados reais (linha com 'source', 'target', 'weight')
        start_row = 0
        for i, row in enumerate(all_values):
            # Verifica se a linha contém as colunas esperadas
            if len(row) >= 2 and "source" in row and "target" in row:
                start_row = i
                break
        
        # Se não achou o cabeçalho
        if start_row == 0 and "source" not in all_values[0]:
             return None 
             
        # Define cabeçalho e dados
        headers = all_values[start_row]
        data = all_values[start_row+1:]
        
        # Cria DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Limpeza: remove linhas vazias e converte peso para número
        df = df[df['source'] != ''].copy() # Garante que tem origem
        if 'weight' in df.columns:
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1)
            
        return df
    except Exception as e:
        st.error(f"Erro ao carregar aba {worksheet.title}: {str(e)}")
        return None

def calcular_comparacao(df1, df2):
    """
    Compara dois DataFrames de arestas e retorna métricas de similaridade.
    """
    # Extrair conjuntos de nós únicos (união de source e target)
    nodes1 = set(df1['source']).union(set(df1['target']))
    nodes2 = set(df2['source']).union(set(df2['target']))
    
    # 1. Índice de Jaccard (Similaridade de Vocabulário)
    interseccao = nodes1.intersection(nodes2)
    uniao = nodes1.union(nodes2)
    jaccard = len(interseccao) / len(uniao) if len(uniao) > 0 else 0
    
    # 2. Diferenças (Análise de Delta)
    exclusivos_antigos = nodes1 - nodes2 # O que foi "esquecido" ou removido
    exclusivos_novos = nodes2 - nodes1   # O que é "novidade"
    
    return {
        "jaccard": jaccard,
        "comuns": sorted(list(interseccao)),
        "exclusivos_antigos": sorted(list(exclusivos_antigos)),
        "exclusivos_novos": sorted(list(exclusivos_novos)),
        "qtd_1": len(nodes1),
        "qtd_2": len(nodes2)
    }

def parse_history_data(all_values):
    """
    Parser robusto para o formato híbrido (Metadata -> Nodes -> Edges).
    Ignora marcadores e cabeçalhos repetidos.
    """
    data = {
        "meta": {},
        "nodes": {},
        "edges": []
    }
    
    current_section = None
    
    for row in all_values:
        if not row: continue
        first_cell = str(row[0]).strip()
        
        # 1. Detecta mudança de seção
        if first_cell == "---METADATA---":
            current_section = "meta"
            continue # Pula a linha do marcador
        elif first_cell == "---NODES---":
            current_section = "nodes"
            continue
        elif first_cell == "---EDGES---":
            current_section = "edges"
            continue
            
        # 2. Ignora cabeçalhos das colunas (se a linha for "Id", "source", etc)
        if first_cell.lower() in ["id", "source", "valor"]:
            continue
            
        # 3. Processa dados
        if current_section == "meta":
            if len(row) >= 2:
                data["meta"][first_cell] = row[1]
                
        elif current_section == "nodes":
            # Esperado: [Id, Freq, Score, Level]
            if len(row) >= 4:
                try:
                    name = row[0]
                    data["nodes"][name] = {
                        "freq": int(row[1]),
                        "score": float(row[2].replace(',', '.')),
                        "level": float(row[3].replace(',', '.'))
                    }
                except: pass
                
        elif current_section == "edges":
            # Esperado: [source, target, weight, salton]
            # Importante: só adiciona se tiver origem e destino
            if len(row) >= 3 and row[1].strip():
                data["edges"].append(row)

    return data

def carregar_grafo_do_sheets(worksheet):
    """
    Carrega o grafo e anexa os metadados ricos ao objeto DataFrame.
    """
    try:
        all_values = worksheet.get_all_values()
        parsed = parse_history_data(all_values)
        
        # Cria DataFrame de arestas (compatibilidade com código antigo)
        if parsed["edges"]:
            df = pd.DataFrame(parsed["edges"], columns=['source', 'target', 'weight', 'salton'])
            # Tratamento de números
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1)
            
            # ANEXA OS DADOS RICOS AO DATAFRAME
            # Isso permite que a gente acesse df.attrs['metadata']['aluno_tema'] na Tab 3!
            df.attrs['nodes_dict'] = parsed['nodes']
            df.attrs['metadata'] = parsed['meta']
            
            return df
        return None
    except Exception as e:
        st.error(f"Erro ao ler aba {worksheet.title}: {e}")
        return None