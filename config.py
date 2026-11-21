import os

from dotenv import load_dotenv

# Cargar variables de entorno: busca el .env en esta carpeta y en las superiores,
# así vale tanto el .env del proyecto como el de la raíz del repositorio.
load_dotenv()

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# data/ guarda todo el estado del agente:
#   chat.db            -> usuarios y chats (SQLite)
#   checkpoints.sqlite -> conversaciones completas de LangGraph
#   chroma/            -> memoria transversal (Chroma)
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, 'chat.db')
# El checkpointer va en su propio fichero: LangGraph crea sus tablas y así la
# interfaz no compite por los bloqueos con los metadatos.
CHECKPOINT_DB = os.path.join(DATA_DIR, 'checkpoints.sqlite')

# Configuración del modelo
DEFAULT_MODEL = 'gpt-4o-mini'
DEFAULT_TEMPERATURE = 0.3
# Mensajes del chat que se envían al modelo en cada turno (contexto inmediato).
MAX_HISTORY_MESSAGES = 20

# Prompt base del asistente. La memoria del usuario se añade aparte, en graph.py.
SYSTEM_PROMPT = (
    'Eres un asistente útil y directo. Respondes en español. '
    'Ten en cuenta la memoria del usuario cuando sea relevante, pero no la '
    'repitas literalmente ni menciones que la tienes.'
)

# Configuración de la memoria transversal
# MAX_MEMORY_RESULTS: cuántos recuerdos se añaden por similitud con el mensaje,
# además de los de CATEGORIAS_SIEMPRE (que se inyectan siempre).
MAX_MEMORY_RESULTS = 3
MEMORY_CATEGORIES = [
    'personal',
    'profesional',
    'preferencias',
    'hechos_importantes',
]
# Categorías cuyos recuerdos activos se inyectan SIEMPRE, sin depender de la
# similitud con el mensaje. Son pocos y son los que definen a la persona: si
# solo se buscara por parecido, una pregunta como "¿a qué me dedico?" podría
# traer el trabajo antiguo (léxicamente cercano) y dejar fuera el vigente.
CATEGORIAS_SIEMPRE = ('personal', 'profesional', 'hechos_importantes')
# Cada cuántos turnos se extrae memoria nueva (1 = en cada respuesta).
EXTRAER_CADA_N_TURNOS = 1

# Configuración de la memoria vectorial (Chroma)
CHROMA_DIR = os.path.join(DATA_DIR, 'chroma')
# Espacio de distancia; debe coincidir con default_space() del embedding.
HNSW_SPACE = 'cosine'
# Distancia por debajo de la cual dos recuerdos se consideran el mismo.
# Depende del modelo de embeddings: hay que calibrarlo, no vale copiarlo.
UMBRAL_DUPLICADO = 0.25

# Configuración del embedding
# large: 3072 dimensiones y más coste; small (1536) suele bastar para frases
# cortas de memoria.
EMBEDDING_MODEL = 'text-embedding-3-small'

# Configuración de la interfaz
PAGE_TITLE = 'Memory Multi-User Chat'
PAGE_ICON = '🤖'


def exigir_api_key() -> str:
    """Devuelve OPENAI_API_KEY o falla con un mensaje claro.

    Se comprueba al usarla y no al importar el módulo: así los tests pueden
    trabajar con un embedding falso sin necesitar credenciales.
    """
    clave = os.getenv('OPENAI_API_KEY')
    if not clave:
        raise RuntimeError(
            'Falta OPENAI_API_KEY: añádela al .env de la raíz del repositorio'
        )
    return clave
