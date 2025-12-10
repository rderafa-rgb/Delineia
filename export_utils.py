import pandas as pd
import io
import json
import networkx as nx
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

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