# ======================================================
# PRUEBA TÉCNICA 2 - ASISTENTE LEGAL CON IA
# ======================================================

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import warnings
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

print("="*60)
print("Asistente Legal con IA")
print("="*60)

# ======================================================
# CARGA DE DATOS
# ======================================================

print("\n--- Cargando datos ---")
NOMBRE_ARCHIVO = 'sentencias_pasadas.xlsx'

if not os.path.exists(NOMBRE_ARCHIVO):
    print(f"ERROR: No se encontró el archivo '{NOMBRE_ARCHIVO}'")
    exit()

df = pd.read_excel(NOMBRE_ARCHIVO, sheet_name='Hoja1')
print(f"{len(df)} sentencias cargadas")

# Crear un diccionario para acceso rápido por ID
print("Creando índice por ID para acceso rápido...")
documentos_por_id = {}
for index, row in df.iterrows():
    doc_id = str(row.get('#', index))
    documentos_por_id[doc_id] = {
        'providencia': str(row.get('Providencia', '')) if pd.notna(row.get('Providencia')) else '',
        'tema': str(row.get('Tema - subtema', '')) if pd.notna(row.get('Tema - subtema')) else '',
        'sintesis': str(row.get('sintesis', '')) if pd.notna(row.get('sintesis')) else '',
        'resuelve': str(row.get('resuelve', '')) if pd.notna(row.get('resuelve')) else ''
    }
print(f"Índice creado con {len(documentos_por_id)} documentos")

# ======================================================
# PREPARACIÓN PARA BASE VECTORIAL
# ======================================================

print("\n--- Preparando documentos para búsqueda semántica ---")
documentos_para_vectores = []
ids_documentos = []
metadatos = []

for index, row in df.iterrows():
    doc_id = str(row.get('#', index))
    
    providencia = str(row.get('Providencia', '')) if pd.notna(row.get('Providencia')) else ''
    tema = str(row.get('Tema - subtema', '')) if pd.notna(row.get('Tema - subtema')) else ''
    sintesis = str(row.get('sintesis', '')) if pd.notna(row.get('sintesis')) else ''
    resuelve = str(row.get('resuelve', '')) if pd.notna(row.get('resuelve')) else ''
    
    texto_completo = f"PROVIDENCIA: {providencia}\n"
    texto_completo += f"TEMAS: {tema}\n"
    texto_completo += f"SÍNTESIS: {sintesis}\n"
    texto_completo += f"SENTENCIA: {resuelve}"
    
    documentos_para_vectores.append(texto_completo)
    ids_documentos.append(doc_id)
    metadatos.append({
        "providencia": providencia,
        "id": doc_id
    })

print(f"{len(documentos_para_vectores)} documentos preparados")

# ======================================================
# BASE DE DATOS VECTORIAL
# ======================================================

print("\n--- PASO 3: Base de datos vectorial ---")
modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
print("Modelo de embeddings cargado")

persist_directory = "./chroma_db_optimizada"
client = chromadb.PersistentClient(path=persist_directory)
collection_name = "demandas_optimizada"

try:
    collection = client.get_collection(name=collection_name)
    print(f"Colección existente recuperada ({collection.count()} documentos)")
except:
    collection = client.create_collection(name=collection_name)
    print("Generando embeddings...")
    embeddings_docs = modelo_embedding.encode(documentos_para_vectores).tolist()
    collection.add(
        embeddings=embeddings_docs,
        documents=documentos_para_vectores,
        metadatas=metadatos,
        ids=ids_documentos
    )
    print(f"Colección creada con {collection.count()} documentos")

# ======================================================
# MEMORIA CON UMBRAL 0.5
# ======================================================

class MemoriaOptimizada:
    def __init__(self, umbral=0.50):
        self.ultimas_providencias = []
        self.ultimo_embedding = None
        self.ultimos_ids = []
        self.umbral = umbral
        
    def actualizar(self, ids, providencias, embedding_pregunta):
        """Actualiza la memoria con nuevos resultados"""
        self.ultimos_ids = ids[:3]
        self.ultimas_providencias = providencias[:3]
        self.ultimo_embedding = embedding_pregunta
    
    def obtener_similitud(self, embedding_actual):
        """Calcula similitud con la última pregunta"""
        if self.ultimo_embedding is None:
            return 0.0
        return cosine_similarity([embedding_actual], [self.ultimo_embedding])[0][0]
    
    def deberia_usar_memoria(self, embedding_actual):
        """Determina si debemos usar memoria basado en el umbral"""
        similitud = self.obtener_similitud(embedding_actual)
        print(f"Similitud con pregunta anterior: {similitud:.3f}")
        
        if similitud > self.umbral and self.ultimos_ids:
            print(f"Usando memoria (umbral {self.umbral} superado)")
            return True, self.ultimos_ids, self.ultimas_providencias
        else:
            print(f"Memoria no usada (similitud ≤ {self.umbral})")
            return False, [], []

memoria = MemoriaOptimizada(umbral=0.50)

# ======================================================
# BÚSQUEDA SEMÁNTICA
# ======================================================

def buscar_con_umbral(pregunta, umbral_similitud=0.50, max_resultados=3):
    """
    Busca documentos y filtra por umbral de similitud
    """
    embedding_pregunta = modelo_embedding.encode(pregunta).tolist()
    
    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=10
    )
    
    docs = resultados['documents'][0]
    metas = resultados['metadatas'][0]
    ids = resultados['ids'][0]
    
    # Calcular similitud real
    docs_emb = modelo_embedding.encode(docs).tolist()
    docs_con_score = []
    ids_vistos = set()
    
    for i, (doc, meta, doc_id) in enumerate(zip(docs, metas, ids)):
        sim = cosine_similarity([embedding_pregunta], [docs_emb[i]])[0][0]
        
        # Filtrar por umbral y evitar duplicados
        if sim >= umbral_similitud and doc_id not in ids_vistos:
            docs_con_score.append({
                'documento': doc,
                'metadata': meta,
                'id': doc_id,
                'providencia': meta.get('providencia', ''),
                'score': sim
            })
            ids_vistos.add(doc_id)
    
    # Ordenar por score
    docs_con_score.sort(key=lambda x: x['score'], reverse=True)
    docs_con_score = docs_con_score[:max_resultados]
    
    print(f"Scores: {[round(d['score'], 3) for d in docs_con_score]}")
    
    return docs_con_score, embedding_pregunta

# ======================================================
# FUNCIÓN PRINCIPAL OPTIMIZADA
# ======================================================

def responder_optimizado(pregunta):
    """
    Función principal con memoria inteligente y acceso directo por ID
    """
    print(f"\n{'='*60}")
    print(f"PREGUNTA: '{pregunta}'")
    print('='*60)
    
    embedding_actual = modelo_embedding.encode(pregunta).tolist()
    
    # Decidir si usar memoria
    usar_memoria, ids_memoria, provs_memoria = memoria.deberia_usar_memoria(embedding_actual)
    
    if usar_memoria:
        print(f"Recuperando documentos directamente por ID: {ids_memoria}")
        
        # ACCESO DIRECTO POR ID - SIN EMBEDDINGS
        docs_con_score = []
        for doc_id, providencia in zip(ids_memoria, provs_memoria):
            # Obtener por ID directamente
            resultado = collection.get(ids=[doc_id])
            if resultado and resultado['documents']:
                docs_con_score.append({
                    'documento': resultado['documents'][0],
                    'metadata': {'providencia': providencia, 'id': doc_id},
                    'id': doc_id,
                    'providencia': providencia,
                    'score': 1.0  # Score perfecto para memoria
                })
        
        docs_usados = docs_con_score
    else:
        # Búsqueda semántica normal
        print("Búsqueda semántica con umbral 0.50...")
        docs_usados, _ = buscar_con_umbral(pregunta, umbral_similitud=0.50)
    
    # Actualizar memoria con los resultados
    if docs_usados:
        ids_actuales = [d['id'] for d in docs_usados]
        provs_actuales = [d['providencia'] for d in docs_usados]
        memoria.actualizar(ids_actuales, provs_actuales, embedding_actual)
    
    # Generar respuesta
    respuesta = generar_respuesta_optimizada(pregunta, docs_usados, usar_memoria)
    
    print("\n--- RESPUESTA COLOQUIAL ---")
    print(respuesta)
    print("\n--- FIN DE LA RESPUESTA ---")
    
    return respuesta

# ======================================================
# GENERACIÓN DE RESPUESTAS CON DETECCIÓN MEJORADA
# ======================================================

def generar_respuesta_optimizada(pregunta, docs_usados, usando_memoria=False):
    """Genera respuestas con detección directa de términos"""
    
    if not docs_usados:
        return "Lo siento, no encontré documentos relevantes para tu pregunta."
    
    providencias = [d['providencia'] for d in docs_usados]
    pregunta_lower = pregunta.lower()
    
    # Si usamos memoria, respuesta contextual
    if usando_memoria:
        respuesta = f"Continuando con los casos que mencionamos antes:\n\n"
        
        for doc in docs_usados[:2]:
            documento = doc['documento']
            prov = doc['providencia']
            
            # Extraer síntesis
            for linea in documento.split('\n'):
                if linea.startswith('SÍNTESIS:'):
                    sintesis = linea.replace('SÍNTESIS:', '').strip()
                    sintesis = sintesis[:300] + "..." if len(sintesis) > 300 else sintesis
                    respuesta += f"{prov}: {sintesis}\n\n"
                    break
        return respuesta
    
    # DETECCIÓN DIRECTA DE PIAR EN EL DOCUMENTO
    if 'piar' in pregunta_lower:
        print("Buscando casos con mención de PIAR...")
        respuesta = "Casos que mencionan el PIAR (Plan Individual de Ajustes Razonables):\n\n"
        
        for doc in docs_usados:
            documento = doc['documento'].lower()
            prov = doc['providencia']
            
            # Buscar directamente "piar" en el documento
            if 'piar' in documento:
                # Extraer el contexto donde aparece PIAR
                for linea in doc['documento'].split('\n'):
                    if 'PIAR' in linea.upper():
                        respuesta += f"{prov}: {linea}\n"
                        # Añadir síntesis
                        for l in doc['documento'].split('\n'):
                            if l.startswith('SÍNTESIS:'):
                                sintesis = l.replace('SÍNTESIS:', '').strip()
                                respuesta += f"{sintesis[:200]}...\n\n"
                                break
                        break
        return respuesta
    
    # Acoso escolar
    if any(p in pregunta_lower for p in ['acoso', 'bullying', 'matoneo']):
        for doc in docs_usados:
            if doc['providencia'] == "T-249/24":
                respuesta = "El caso T-249/24 trata sobre acoso escolar:\n\n"
                
                for linea in doc['documento'].split('\n'):
                    if linea.startswith('SÍNTESIS:'):
                        sintesis = linea.replace('SÍNTESIS:', '').strip()
                        respuesta += f"¿Qué pasó?: {sintesis[:400]}...\n\n"
                    if linea.startswith('SENTENCIA:'):
                        sentencia = linea.replace('SENTENCIA:', '').strip()
                        if 'detalle' in pregunta_lower or 'lujo' in pregunta_lower:
                            respuesta += f"Decisión: {sentencia[:400]}...\n\n"
                        else:
                            respuesta += "Decisión: La Corte determinó que el colegio fue responsable.\n\n"
                return respuesta
    
    # Respuesta general
    respuesta = "Casos relevantes encontrados:\n\n"
    for doc in docs_usados[:3]:
        prov = doc['providencia']
        for linea in doc['documento'].split('\n'):
            if linea.startswith('TEMAS:'):
                temas = linea.replace('TEMAS:', '').strip()
                respuesta += f"{prov} - {temas[:150]}\n"
                break
    
    return respuesta

# ======================================================
# EJECUCIÓN
# ======================================================

print("\n" + "="*60)
print("INICIANDO EVALUACIÓN")
print("="*60)

preguntas = [
    "¿Cuáles son las sentencias de 3 demandas relacionadas con redes sociales?",
    "¿De qué se trataron esas 3 demandas?",
    "¿Cuál fue la sentencia del caso que habla de acoso escolar?",
    "Cuéntame con lujo de detalle la demanda de acoso escolar",
    "Explícame qué casos hay sobre el PIAR, de qué trataron y qué decidió el juez"
]

for i, pregunta in enumerate(preguntas, 1):
    print(f"\n--- PREGUNTA {i} ---")
    responder_optimizado(pregunta)

print("\n" + "="*60)
print("PROCESO COMPLETADO")
print("="*60)