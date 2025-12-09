import pandas as pd
from io import BytesIO

def safe_get(d, keys, default=''):
    """
    Navega com segurança por dicionários aninhados, evitando erros com None.
    Uso: safe_get(artigo, ['primary_location', 'source', 'display_name'])
    """
    if not isinstance(d, dict):
        return default
    
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
            
        if current is None:
            return default
            
    return current if current else default

def clean_authors(authors_list):
    """Transforma a lista de autores em string legível."""
    if not authors_list or not isinstance(authors_list, list):
        return ""
    
    names = []
    for auth in authors_list:
        if isinstance(auth, dict):
            # Tenta pegar author -> display_name
            name = safe_get(auth, ['author', 'display_name'])
            if name:
                names.append(name)
                
    return ", ".join(names)

def format_concepts_detailed(concepts_list):
    """Formata conceitos com seus atributos para o Excel."""
    if not concepts_list or not isinstance(concepts_list, list):
        return ""
    
    # Ordena por score
    sorted_concepts = sorted(concepts_list, key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0, reverse=True)
    
    formatted_items = []
    for c in sorted_concepts:
        if not isinstance(c, dict): continue
        
        name = c.get('display_name', '')
        score = c.get('score', 0)
        level = c.get('level', '?')
        
        if name:
            formatted_items.append(f"{name} (S:{score:.3f}, L:{level})")
            
    return "; ".join(formatted_items)

def get_concept_names(concepts_list):
    """Retorna lista simples de nomes para BibTeX/RIS."""
    if not concepts_list or not isinstance(concepts_list, list):
        return []
    
    sorted_concepts = sorted(concepts_list, key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0, reverse=True)
    return [c.get('display_name', '') for c in sorted_concepts if isinstance(c, dict) and c.get('display_name')]

def prepare_dataframe(results):
    """Prepara DataFrame incluindo a coluna rica de Conceitos."""
    if not results:
        return pd.DataFrame()
    
    data = []
    for r in results:
        # Extração segura usando a função auxiliar
        doi_raw = r.get('doi')
        doi_safe = doi_raw if doi_raw else ''
        
        titulo = r.get('title') or 'Sem Título'
        ano = r.get('publication_year') or ''
        data_pub = r.get('publication_date') or ''
        tipo = r.get('type') or ''
        citacoes = r.get('cited_by_count') or 0
        
        # Extração complexa segura
        autores = clean_authors(r.get('authorships'))
        revista = safe_get(r, ['primary_location', 'source', 'display_name'])
        conceitos = format_concepts_detailed(r.get('concepts'))
        abstract = "Disponível (Índice Invertido)" if r.get('abstract_inverted_index') else "Não disponível"

        row = {
            'ID OpenAlex': r.get('id', ''),
            'Título': titulo,
            'Ano': ano,
            'Data Publicação': data_pub,
            'Autores': autores,
            'Revista/Fonte': revista,
            'Conceitos (Score/Level)': conceitos,
            'Tipo': tipo,
            'Citações': citacoes,
            'DOI': doi_safe,
            'Abstract': abstract
        }
        data.append(row)
    
    return pd.DataFrame(data)

def generate_excel(results):
    """Gera Excel com a nova coluna de conceitos."""
    df = prepare_dataframe(results)
    output = BytesIO()
    
    # Se o dataframe estiver vazio, retorna buffer vazio mas válido
    if df.empty:
        df = pd.DataFrame({'Info': ['Nenhum dado encontrado para exportar']})

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados Delinéia')
        worksheet = writer.sheets['Resultados Delinéia']
        
        # Ajustes de largura
        worksheet.set_column('B:B', 40) # Título
        worksheet.set_column('E:E', 30) # Autores
        worksheet.set_column('F:F', 25) # Revista
        worksheet.set_column('G:G', 50) # Conceitos
        
    return output.getvalue()

def generate_bibtex(results):
    """Gera BibTeX incluindo conceitos como keywords (Versão Blindada)."""
    if not results:
        return ""
        
    bibtex_str = ""
    for r in results:
        authors = clean_authors(r.get('authorships'))
        
        # Autor seguro para a chave
        if authors:
            first_author_raw = authors.split(',')[0]
            # Pega apenas o último sobrenome
            first_author = first_author_raw.split(' ')[-1]
        else:
            first_author = "Unknown"
            
        # Limpa caracteres não alfanuméricos da chave
        first_author = "".join(filter(str.isalnum, first_author))
        
        year = str(r.get('publication_year') or '0000')
        cite_key = f"{first_author}{year}"
        
        # Tratamento seguro de DOI
        doi_raw = r.get('doi')
        doi_clean = doi_raw.replace('https://doi.org/', '') if doi_raw else ""
        
        # Título e Revista seguros
        title = r.get('title') or "No Title"
        journal = safe_get(r, ['primary_location', 'source', 'display_name'])
        
        keywords_list = get_concept_names(r.get('concepts'))
        keywords_str = ", ".join(keywords_list)
        
        bibtex_str += f"@article{{{cite_key},\n"
        bibtex_str += f"  author = {{{authors.replace(', ', ' and ')}}},\n"
        bibtex_str += f"  title = {{{title}}},\n"
        if journal:
            bibtex_str += f"  journal = {{{journal}}},\n"
        bibtex_str += f"  year = {{{year}}},\n"
        if doi_clean:
            bibtex_str += f"  doi = {{{doi_clean}}},\n"
        if keywords_str:
            bibtex_str += f"  keywords = {{{keywords_str}}},\n"
        bibtex_str += "}\n\n"
    return bibtex_str

def generate_ris(results):
    """Gera RIS com keywords para cada conceito (Versão Blindada)."""
    if not results:
        return ""
        
    ris_str = ""
    for r in results:
        ris_str += "TY  - JOUR\n"
        
        title = r.get('title') or "No Title"
        ris_str += f"TI  - {title}\n"
        
        # Autores
        raw_authors = r.get('authorships')
        if isinstance(raw_authors, list):
            for auth in raw_authors:
                name = safe_get(auth, ['author', 'display_name'])
                if name:
                    ris_str += f"AU  - {name}\n"
        
        year = r.get('publication_year') or ''
        ris_str += f"PY  - {year}\n"
        
        journal = safe_get(r, ['primary_location', 'source', 'display_name'])
        if journal:
            ris_str += f"JO  - {journal}\n"
        
        doi_raw = r.get('doi')
        if doi_raw:
            doi_clean = doi_raw.replace('https://doi.org/', '')
            ris_str += f"DO  - {doi_clean}\n"
            ris_str += f"UR  - {doi_raw}\n"
        
        keywords_list = get_concept_names(r.get('concepts'))
        for kw in keywords_list:
            ris_str += f"KW  - {kw}\n"
            
        ris_str += "ER  - \n\n"
    return ris_str