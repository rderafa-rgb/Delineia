# -*- coding: utf-8 -*-
"""
TESTE DE DIAGNÓSTICO DO GEMINI - TERMINAL
==========================================
Execute este script no terminal do VSCode para diagnosticar
o problema com o Gemini API.

USO:
    python teste_gemini_terminal.py

Certifique-se de que o arquivo .streamlit/secrets.toml existe
ou defina a variável de ambiente GEMINI_API_KEY
"""

import os
import sys
import time

print("=" * 70)
print("🔍 DIAGNÓSTICO DO GEMINI API - VERSÃO TERMINAL")
print("=" * 70)
print()

# ============ ETAPA 1: Verificar API Key ============
print("📌 ETAPA 1: Verificando API Key...")
print("-" * 50)

api_key = None

# Método 1: Variável de ambiente
api_key = os.environ.get("GEMINI_API_KEY", "")
if api_key:
    print(f"✅ Encontrada via VARIÁVEL DE AMBIENTE")
    print(f"   Tamanho: {len(api_key)} caracteres")
    print(f"   Início: {api_key[:15]}...")

# Método 2: Arquivo secrets.toml
if not api_key:
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        print(f"📄 Lendo {secrets_path}...")
        try:
            with open(secrets_path, 'r') as f:
                content = f.read()
                # Procurar GEMINI_API_KEY
                import re
                match = re.search(r'GEMINI_API_KEY\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    api_key = match.group(1)
                    print(f"✅ Encontrada no secrets.toml")
                    print(f"   Tamanho: {len(api_key)} caracteres")
                    print(f"   Início: {api_key[:15]}...")
                else:
                    print(f"❌ GEMINI_API_KEY não encontrada no arquivo")
                    print(f"   Conteúdo do arquivo (primeiras linhas):")
                    for line in content.split('\n')[:10]:
                        if 'KEY' in line.upper() or 'GEMINI' in line.upper():
                            print(f"      {line[:60]}...")
        except Exception as e:
            print(f"❌ Erro ao ler arquivo: {e}")
    else:
        print(f"❌ Arquivo {secrets_path} não existe")

# Método 3: Hardcoded para teste (REMOVER EM PRODUÇÃO!)
if not api_key:
    print()
    print("⚠️  API Key não encontrada automaticamente.")
    print("   Você pode:")
    print("   1. Criar .streamlit/secrets.toml com GEMINI_API_KEY = \"sua_key\"")
    print("   2. Definir variável: export GEMINI_API_KEY=\"sua_key\"")
    print("   3. Colar a key aqui para teste rápido:")
    print()
    api_key = input("   Cole a GEMINI_API_KEY (ou Enter para sair): ").strip()
    if not api_key:
        print("\n❌ Sem API Key, não é possível continuar.")
        sys.exit(1)

print()

# ============ ETAPA 2: Instalar/Importar biblioteca ============
print("📌 ETAPA 2: Verificando biblioteca google-generativeai...")
print("-" * 50)

try:
    import google.generativeai as genai
    print(f"✅ Biblioteca importada com sucesso")
    print(f"   Versão: {genai.__version__ if hasattr(genai, '__version__') else 'N/A'}")
except ImportError:
    print("❌ Biblioteca não instalada. Instalando...")
    os.system("pip install google-generativeai")
    import google.generativeai as genai
    print("✅ Instalada e importada")

print()

# ============ ETAPA 3: Configurar API ============
print("📌 ETAPA 3: Configurando API...")
print("-" * 50)

try:
    genai.configure(api_key=api_key)
    print("✅ genai.configure() executado sem erros")
except Exception as e:
    print(f"❌ Erro ao configurar: {type(e).__name__}: {e}")
    sys.exit(1)

print()

# ============ ETAPA 4: Criar modelo ============
print("📌 ETAPA 4: Criando modelo...")
print("-" * 50)

try:
    model = genai.GenerativeModel(
        'gemini-2.5-pro',
        generation_config={
            'temperature': 1.2,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 8192,
        }
    )
    print(f"✅ Modelo criado: {model.model_name}")
except Exception as e:
    print(f"❌ Erro ao criar modelo: {type(e).__name__}: {e}")
    sys.exit(1)

print()

# ============ ETAPA 5: Teste simples ============
print("📌 ETAPA 5: Teste de geração SIMPLES...")
print("-" * 50)

prompt_simples = "Responda apenas com 'OK, funcionando!' sem nada mais."

print(f"   Prompt: {prompt_simples}")
print(f"   Aguardando resposta...")

try:
    start = time.time()
    response = model.generate_content(prompt_simples)
    elapsed = time.time() - start
    
    print(f"✅ Resposta recebida em {elapsed:.2f}s")
    
    # Verificar prompt_feedback
    if hasattr(response, 'prompt_feedback'):
        print(f"   Prompt feedback: {response.prompt_feedback}")
    
    # Tentar extrair texto
    if hasattr(response, 'text'):
        try:
            text = response.text
            print(f"✅ Texto extraído: '{text[:100]}'")
        except ValueError as e:
            print(f"❌ Erro ao extrair .text: {e}")
            
            # Verificar candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                print(f"   Candidate finish_reason: {getattr(candidate, 'finish_reason', 'N/A')}")
                
                if hasattr(candidate, 'safety_ratings'):
                    print("   Safety ratings:")
                    for rating in candidate.safety_ratings:
                        cat = getattr(rating, 'category', 'N/A')
                        prob = getattr(rating, 'probability', 'N/A')
                        print(f"      - {cat}: {prob}")
            else:
                print(f"   Candidates: {response.candidates}")
    else:
        print(f"⚠️ Response não tem atributo .text")
        print(f"   Type: {type(response)}")
        print(f"   Dir: {[a for a in dir(response) if not a.startswith('_')]}")

except Exception as e:
    print(f"❌ ERRO na geração: {type(e).__name__}")
    print(f"   Mensagem: {e}")
    import traceback
    traceback.print_exc()

print()

# ============ ETAPA 6: Teste com prompt real ============
print("📌 ETAPA 6: Teste com prompt REAL do Delinéia...")
print("-" * 50)

prompt_real = """Você é um professor universitário experiente orientando um aluno de pesquisa.

**CONTEXTO DO PROJETO:**

Aluno: Maria Silva (você vai chamá-la de Maria)
Tema proposto: Inteligência Artificial na Educação
Questão de pesquisa: Como a IA pode apoiar o ensino de matemática?
Palavras-chave escolhidas: IA, Educação, Matemática, Ensino

---

**SUA TAREFA:**

Escreva DOIS parágrafos conversando com Maria:

**PARÁGRAFO 1 - Sobre as palavras-chave:**
• Comece com: "Maria, as palavras-chave que você designou para o projeto..."
• Comente especificamente sobre as palavras-chave escolhidas

**PARÁGRAFO 2 - Sobre a questão de pesquisa:**
• Comente sobre a questão de pesquisa
• Encerre com: "Recomendo que você converse com seu orientador sobre esses pontos."

Escreva agora:"""

print(f"   Tamanho do prompt: {len(prompt_real)} caracteres")
print(f"   Aguardando resposta (pode demorar 30-60s)...")

try:
    start = time.time()
    response = model.generate_content(prompt_real)
    elapsed = time.time() - start
    
    print(f"✅ Resposta recebida em {elapsed:.2f}s")
    
    # Extrair texto
    text = None
    
    if hasattr(response, 'text'):
        try:
            text = response.text
        except:
            pass
    
    if not text and hasattr(response, 'candidates') and response.candidates:
        try:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts
                if parts:
                    text = parts[0].text
        except:
            pass
    
    if text:
        print()
        print("=" * 70)
        print("📝 RESPOSTA DO GEMINI:")
        print("=" * 70)
        print(text)
        print("=" * 70)
        print()
        print(f"✅ SUCESSO! O Gemini está funcionando!")
        print(f"   Tamanho da resposta: {len(text)} caracteres")
    else:
        print(f"❌ Não foi possível extrair texto da resposta")
        print(f"   Response type: {type(response)}")
        if hasattr(response, 'candidates'):
            print(f"   Candidates: {response.candidates}")

except Exception as e:
    print(f"❌ ERRO na geração: {type(e).__name__}")
    print(f"   Mensagem: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("🏁 DIAGNÓSTICO CONCLUÍDO")
print("=" * 70)
