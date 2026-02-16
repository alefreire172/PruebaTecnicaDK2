# Prueba Técnica 2 - Asistente Legal con IA

## Descripción
Sistema RAG (Retrieval-Augmented Generation) para consulta de sentencias 
de la Corte Constitucional colombiana.

## Datos procesados
- 329 sentencias cargadas desde `sentencias_pasadas.xlsx`
- Base de datos vectorial ChromaDB persistente
- Modelo de embeddings: all-MiniLM-L6-v2

## Funcionalidades implementadas
- Búsqueda semántica con umbral de similitud 0.50
- Memoria conversacional inteligente
- Acceso directo por ID (optimizado)
- Detección de PIAR en texto completo
- Respuestas en lenguaje coloquial

## Resultados obtenidos
| Pregunta | Casos encontrados                   |
|----------|-------------------------------------|
| Redes sociales | T-394/24, T-063/24            |
| Acoso escolar  | T-249/24                      |
| PIAR           | T-249/24 (detección en texto) |

## Instalación y ejecución
```bash
pip install pandas openpyxl sentence-transformers chromadb scikit-learn
python codigo_final.py
