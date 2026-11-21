"""Grafo del chat: memoria de contexto + memoria transversal.

Tres nodos, uno por preocupación:

    aprender  ->  recuperar  ->  responder

- ``aprender`` extrae hechos duraderos del último intercambio y los guarda en
  la memoria transversal. Va **antes** de responder para que un dato que el
  usuario acaba de contar (un cambio de empleo, una mudanza) esté ya en la
  memoria que se inyecta en ese mismo turno, y no en el siguiente. Al extractor
  se le pasan los recuerdos que ya existen, **agrupados por categoría**, para
  que indique cuáles quedan obsoletos (`sustituye_a`, una lista de ids). Un
  hecho que ya está guardado no se vuelve a procesar, aunque el extractor
  insista: así no puede desactivar por error un recuerdo que no venía a cuento.
  No ve la respuesta del asistente: no la necesita y así no se retroalimenta de
  sus propios resúmenes.
- ``recuperar`` reúne los recuerdos que se van a inyectar: **siempre** los
  personales y profesionales activos (que definen a la persona y no pueden
  depender del parecido con la pregunta), más los que más tienen que ver con lo
  que acaba de escribir, y los formatea.
- ``responder`` monta el prompt (rol + memoria + últimos mensajes) y llama al
  modelo. La respuesta se añade al estado, que es la memoria de contexto: el
  checkpointer guarda un hilo por chat, con ``thread_id = id del chat``.

El mensaje de sistema **no** se guarda en el estado: se compone en cada turno, así
la memoria que se inyecta siempre está al día y no quedan copias obsoletas en el
historial.

El recorte de contexto (``MAX_HISTORY_MESSAGES``) se aplica solo al construir el
prompt; el estado conserva la conversación completa para poder reabrir el chat.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, SecretStr

import memory_manager
from config import (
    CHECKPOINT_DB,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    EXTRAER_CADA_N_TURNOS,
    MAX_HISTORY_MESSAGES,
    MEMORY_CATEGORIES,
    SYSTEM_PROMPT,
    exigir_api_key,
)

PROMPT_EXTRACCION = """\
Extraes datos duraderos sobre el usuario a partir de un trozo de conversación.

Reglas generales:
- Solo hechos estables y útiles en futuras conversaciones: cómo se llama, dónde
  vive, en qué trabaja, qué prefiere, circunstancias importantes de su vida.
- El nombre NO es un saludo: "me llamo José" o "mi nombre es José" es un hecho
  estable. Guárdalo como "El usuario se llama José.".
- Nunca lo que dijo el asistente, ni preguntas del usuario.
- Nada pasajero: saludos, lo que está haciendo ahora, el tiempo, estados de ánimo.
- Escribe cada hecho en tercera persona y empieza SIEMPRE por "El usuario".
  No uses el nombre propio: así el mismo hecho se guarda siempre con las mismas
  palabras y se puede comparar.
- Una sola frase breve por hecho.

Recibirás los RECUERDOS ACTUALES agrupados por categoría, cada uno con su id.

Hechos que ocupan un hueco único (sustitución OBLIGATORIA)
----------------------------------------------------------
Hay hechos que solo pueden tener un valor vigente a la vez: el empleo actual,
la ciudad donde vive, su estado civil, su pareja, su mascota, su vehículo...
Cuando el hecho nuevo ocupa el mismo hueco que uno o varios de los RECUERDOS
ACTUALES, los anteriores quedan obsoletos aunque el usuario no diga "ya no".
En ese caso rellena `sustituye_a` con la LISTA de ids de los recuerdos que
quedan obsoletos. Es obligatorio.

Ejemplos:
- "Trabajo en una empresa de logística" y más tarde "me han aceptado para un
  nuevo empleo de controlador aéreo" -> el de controlador aéreo sustituye al
  de logística.
- Si además existían "lleva la zona de Italia" y "lleva la zona de Francia",
  esos hechos dependían del empleo anterior: incluye también sus ids, porque
  dejan de ser ciertos al cambiar de trabajo.
- "Vivo en Madrid" y más tarde "me he mudado a Valencia" -> el de Valencia
  sustituye al de Madrid.
- "Estoy soltero" y más tarde "me caso en junio" -> el de casado sustituye al
  de soltero.

Hechos que se suman (NO se sustituyen)
--------------------------------------
Otros hechos conviven sin problema: aficiones, idiomas, familiares, gustos
distintos. Que el usuario mencione un gusto nuevo NO invalida los anteriores.

- "Trabajo en logística" y "me encargo de la zona de Italia" describen el
  MISMO empleo: se suman. No sustituyas el empleo por un detalle suyo.
- En `profesional` solo hay sustitución cuando cambia el empleo o el puesto de
  verdad (nuevo trabajo, ascenso, despido, jubilación), no cuando se añaden
  zonas, clientes o tareas del mismo trabajo.

En `preferencias` solo se sustituye si el usuario dice claramente que ha
cambiado de opinión: "ya no me gusta", "ahora prefiero", "he dejado de",
"en vez de", "en lugar de". Si no lo dice, el hecho nuevo se suma.

Reglas de `sustituye_a`:
- Es una lista de ids EXACTOS de RECUERDOS ACTUALES; vacía si no hay que
  sustituir nada.
- Solo se pueden sustituir recuerdos de la MISMA categoría que el hecho nuevo.
- Si el hecho ya está guardado con el mismo significado, NO lo devuelvas, ni
  siquiera para "actualizarlo": no cambia nada.

Si no hay nada nuevo ni relevante, devuelve la lista vacía.
"""


class RecuerdoExtraido(BaseModel):
    """Un hecho que el extractor propone guardar."""

    # Debe coincidir con MEMORY_CATEGORIES de config.py.
    categoria: Literal['personal', 'profesional', 'preferencias', 'hechos_importantes']
    hecho: str = Field(min_length=1)
    # Ids de los recuerdos que este hecho deja obsoletos; vacío si es nuevo.
    sustituye_a: list[str] = Field(default_factory=list)


class Recuerdos(BaseModel):
    """Salida completa del extractor."""

    recuerdos: list[RecuerdoExtraido] = Field(default_factory=list)


class EstadoChat(TypedDict):
    """Estado que viaja por el grafo y se guarda en el checkpointer."""

    mensajes: Annotated[list[BaseMessage], add_messages]
    usuario: str  # slug del usuario
    chat: str  # id del chat (thread_id)
    memoria: str  # bloque de memoria recuperada para este turno


def crear_llm(temperatura: float = DEFAULT_TEMPERATURE) -> ChatOpenAI:
    """Crea el modelo configurado en `config.py`."""
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=temperatura,
        api_key=SecretStr(exigir_api_key()),
    )


# ---------------------------------------------------------------------------
# Utilidades de los nodos
# ---------------------------------------------------------------------------
def _toca_extraer(mensajes: list[BaseMessage]) -> bool:
    """Decide si en este turno hay que extraer memoria.

    Con ``EXTRAER_CADA_N_TURNOS = 1`` se extrae siempre; con 3, uno de cada tres
    mensajes del usuario.
    """
    if EXTRAER_CADA_N_TURNOS <= 1:
        return True
    turnos = sum(1 for mensaje in mensajes if isinstance(mensaje, HumanMessage))
    return turnos % EXTRAER_CADA_N_TURNOS == 0


def _ultimo_intercambio(mensajes: list[BaseMessage]) -> str:
    """Texto de los últimos mensajes, para pasárselo al extractor.

    Como ``aprender`` va antes de responder, en el turno actual solo está el
    mensaje del usuario; los anteriores pueden incluir respuestas del asistente,
    que el prompt manda ignorar.
    """
    lineas: list[str] = []
    for mensaje in mensajes[-4:]:
        if isinstance(mensaje, HumanMessage) and isinstance(mensaje.content, str):
            lineas.append(f'Usuario: {mensaje.content}')
        elif isinstance(mensaje, AIMessage) and isinstance(mensaje.content, str):
            lineas.append(f'Asistente: {mensaje.content}')
    return '\n'.join(lineas)


def _ultimo_mensaje_humano(mensajes: list[BaseMessage]) -> str:
    """Texto del último mensaje del usuario (cadena vacía si no hay)."""
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, HumanMessage) and isinstance(mensaje.content, str):
            return mensaje.content
    return ''


def _normalizar(texto: str) -> str:
    """Forma canónica de un hecho, para compararlo consigo mismo.

    Solo unifica mayúsculas y espacios: no se toca la puntuación porque el id
    de Chroma tampoco lo hace. Sirve para detectar que el extractor está
    repitiendo un hecho que ya está guardado.
    """
    return ' '.join(texto.casefold().split())


# Fórmulas con las que alguien dice su propio nombre. Solo estas: no se acepta
# "soy X" porque también sirve para profesiones ("soy controlador aéreo").
PATRON_NOMBRE = re.compile(
    r'\b(?i:me llamo|mi nombre es)\s+'
    r'([\wáéíóúüñ]+(?:\s+[\wáéíóúüñ]+)?)'
)

# Palabras que cortan el nombre capturado: "me llamo José y trabajo..." no
# debe guardar "José y trabajo" como nombre.
CORTES_NOMBRE = frozenset(
    {
        'y', 'e', 'o', 'u', 'pero', 'aunque', 'que', 'como', 'de', 'del', 'la',
        'el', 'los', 'las', 'un', 'una', 'en', 'me', 'mi', 'soy', 'estoy',
        'vivo', 'trabajo', 'tengo', 'quiero', 'puedo', 'necesito', 'para',
        'por', 'con', 'sin', 'sobre', 'también', 'tambien', 'además', 'ademas',
        'otra', 'vez', 'encantado', 'encantada', 'gusto', 'mucho', 'mucha',
        'aquí', 'ahora',
    }
)

# Marcas de cambio de opinión explícito. En `preferencias` no se sustituye sin
# una de ellas: un gusto nuevo se suma, no reemplaza al anterior.
MARCAS_CAMBIO_PREFERENCIAS = (
    'ya no',
    'he dejado de',
    'he cambiado',
    'ahora prefiero',
    'ahora me gusta',
    'en vez de',
    'en lugar de',
)


def _permite_sustituir(categoria: str, hecho: str, transcripcion: str) -> bool:
    """Indica si ese hecho puede sustituir recuerdos de su categoría.

    En `preferencias` el extractor tiende a tratar un gusto nuevo como si
    reemplazara al anterior (viajar -> gastronomía). Solo se acepta la
    sustitución si hay una marca explícita de cambio de opinión; si no, el
    hecho se suma y `fusionar` se encarga de los duplicados.
    """
    if categoria != 'preferencias':
        return True
    texto = f'{hecho} {transcripcion}'.casefold()
    return any(marca in texto for marca in MARCAS_CAMBIO_PREFERENCIAS)


def _nombre_declarado(texto: str) -> str:
    """Nombre que el usuario dice de sí mismo, o cadena vacía.

    Red determinista para un dato que el extractor a veces clasifica como
    saludo pasajero. Solo reconoce las fórmulas explícitas «me llamo» y «mi
    nombre es»; el resto de la frase no puede colarse porque el nombre se corta
    en la primera palabra funcional ("José y trabajo..." -> "José").
    """
    coincidencia = PATRON_NOMBRE.search(texto)
    if not coincidencia:
        return ''

    palabras: list[str] = []
    for palabra in coincidencia.group(1).split():
        if palabra.casefold() in CORTES_NOMBRE:
            break
        palabras.append(palabra[:1].upper() + palabra[1:])
    return ' '.join(palabras)


def _guardar_nombre(
    slug: str,
    chat: str,
    mensajes: list[BaseMessage],
    existentes: list[memory_manager.Recuerdo],
) -> None:
    """Guarda el nombre declarado por el usuario, sin depender del extractor.

    Si ya hay un nombre activo distinto, lo sustituye (el usuario puede estar
    corrigiéndolo). Si ya está, no hace nada. Los errores se ignoran: la
    memoria es una mejora, no un requisito para responder.
    """
    nombre = _nombre_declarado(_ultimo_mensaje_humano(mensajes))
    if not nombre:
        return

    hecho = f'El usuario se llama {nombre}.'
    personales = [
        recuerdo for recuerdo in existentes if recuerdo.categoria == 'personal'
    ]
    if any(_normalizar(recuerdo.texto) == _normalizar(hecho) for recuerdo in personales):
        return

    for recuerdo in personales:
        if 'se llama' in _normalizar(recuerdo.texto):
            try:
                memory_manager.sustituir_varios(
                    slug,
                    [recuerdo.id],
                    hecho,
                    categoria='personal',
                    origen_chat=chat,
                )
            except Exception:  # noqa: BLE001 - no romper el turno
                return
            return

    try:
        memory_manager.agregar(slug, 'personal', hecho, chat)
    except Exception:  # noqa: BLE001 - idem
        return


def _catalogo(recuerdos: list[memory_manager.Recuerdo]) -> str:
    """Recuerdos activos agrupados por categoría, con su id delante.

    La lista plana anterior ocultaba qué hechos competían por el mismo hueco
    (por ejemplo, dos hechos "profesional" contradictorios). Agrupados, el
    extractor puede razonar por categoría antes de decidir una sustitución.
    """
    por_categoria: dict[str, list[memory_manager.Recuerdo]] = {}
    for recuerdo in recuerdos:
        por_categoria.setdefault(recuerdo.categoria, []).append(recuerdo)

    lineas: list[str] = []
    for categoria in MEMORY_CATEGORIES:
        lista = por_categoria.pop(categoria, [])
        if lista:
            lineas.append(f'[{categoria}]')
            lineas.extend(f'{recuerdo.id} | {recuerdo.texto}' for recuerdo in lista)
    # Categorías desconocidas (datos antiguos): se muestran al final.
    for categoria, lista in por_categoria.items():
        lineas.append(f'[{categoria}]')
        lineas.extend(f'{recuerdo.id} | {recuerdo.texto}' for recuerdo in lista)

    return '\n'.join(lineas) or '(sin recuerdos)'


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------
def construir(
    *,
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Monta y compila el grafo.

    Args:
        llm: modelo a usar; en las pruebas se inyecta un doble para no llamar a
            la API. Sirve tanto para responder como para extraer.
        checkpointer: persistencia del hilo; sin él no hay memoria de contexto.
    """
    modelo = llm or crear_llm()
    extractor = llm or crear_llm(temperatura=0.0)
    estructurado = extractor.with_structured_output(Recuerdos)

    def nodo_recuperar(state: EstadoChat) -> dict[str, Any]:
        consulta = _ultimo_mensaje_humano(state['mensajes'])
        if not consulta:
            return {'memoria': ''}
        # No es un top-k puro: los hechos personales y profesionales se
        # incluyen siempre, para que una pregunta como "¿a qué me dedico?" no
        # se quede con el trabajo antiguo solo porque se parece más al texto.
        recuerdos = memory_manager.recuperar_contexto(state['usuario'], consulta)
        return {'memoria': memory_manager.formatear(recuerdos)}

    def nodo_responder(state: EstadoChat) -> dict[str, Any]:
        sistema = SYSTEM_PROMPT
        if state['memoria']:
            sistema = f'{SYSTEM_PROMPT}\n\n{state["memoria"]}'
        # Solo se recorta lo que se manda al modelo; el estado guarda todo.
        recientes = state['mensajes'][-MAX_HISTORY_MESSAGES:]
        respuesta = modelo.invoke([SystemMessage(content=sistema), *recientes])
        return {'mensajes': [respuesta]}

    def nodo_aprender(state: EstadoChat) -> dict[str, Any]:
        if not _toca_extraer(state['mensajes']):
            return {}
        transcripcion = _ultimo_intercambio(state['mensajes'])
        if not transcripcion:
            return {}

        # Se le pasan los recuerdos que ya tiene para que él decida cuáles
        # repiten o contradicen, en lugar de fiarlo todo a una distancia.
        try:
            existentes = memory_manager.listar(state['usuario'])
        except Exception:  # noqa: BLE001 - sin memoria se sigue igual
            existentes = []

        # Red determinista para el nombre. Se hace antes de construir el
        # catálogo para que el extractor lo vea ya guardado y no lo duplique.
        _guardar_nombre(state['usuario'], state['chat'], state['mensajes'], existentes)
        try:
            existentes = memory_manager.listar(state['usuario'])
        except Exception:  # noqa: BLE001 - se sigue con la lista anterior
            pass

        catalogo = _catalogo(existentes[:50])

        try:
            extraidos = cast(
                Recuerdos,
                estructurado.invoke(
                    [
                        SystemMessage(content=PROMPT_EXTRACCION),
                        HumanMessage(
                            content=(
                                f'RECUERDOS ACTUALES\n{catalogo}\n\n'
                                f'CONVERSACIÓN\n{transcripcion}'
                            )
                        ),
                    ]
                ),
            )
        except Exception:  # noqa: BLE001 - aprender no puede romper la respuesta
            # Si el extractor falla, la conversación sigue: la memoria es una
            # mejora, no un requisito para contestar.
            return {}

        por_id = {recuerdo.id: recuerdo for recuerdo in existentes}
        # Hechos que ya están guardados, en forma canónica.
        ya_guardados = {_normalizar(recuerdo.texto) for recuerdo in existentes}
        pendientes: list[memory_manager.Candidato] = []

        for recuerdo in extraidos.recuerdos:
            # Si el hecho ya está en la memoria no se hace NADA, aunque el
            # extractor lo devuelva. Sin esta guarda se dio el fallo real: el
            # modelo reemitió un hecho existente y apuntó `sustituye_a` a un
            # recuerdo distinto (Francia), desactivándolo sin motivo.
            if _normalizar(recuerdo.hecho) in ya_guardados:
                continue

            # El modelo puede inventarse ids o apuntar a otra categoría: solo
            # se aceptan recuerdos reales de la MISMA categoría, sin repetir.
            objetivos: list[memory_manager.Recuerdo] = []
            vistos: set[str] = set()
            for identificador in recuerdo.sustituye_a:
                objetivo = por_id.get(identificador)
                if (
                    objetivo is not None
                    and objetivo.categoria == recuerdo.categoria
                    and objetivo.id not in vistos
                ):
                    vistos.add(objetivo.id)
                    objetivos.append(objetivo)

            # Un cambio puede invalidar varios recuerdos a la vez (el empleo
            # anterior y las zonas que llevaba). En `preferencias` hace falta
            # además una marca explícita de cambio de opinión.
            if objetivos and _permite_sustituir(
                recuerdo.categoria, recuerdo.hecho, transcripcion
            ):
                try:
                    memory_manager.sustituir_varios(
                        state['usuario'],
                        [objetivo.id for objetivo in objetivos],
                        recuerdo.hecho,
                        categoria=recuerdo.categoria,
                        origen_chat=state['chat'],
                    )
                except ValueError:
                    pendientes.append(
                        memory_manager.Candidato(
                            categoria=recuerdo.categoria,
                            texto=recuerdo.hecho,
                            origen_chat=state['chat'],
                        )
                    )
            else:
                pendientes.append(
                    memory_manager.Candidato(
                        categoria=recuerdo.categoria,
                        texto=recuerdo.hecho,
                        origen_chat=state['chat'],
                    )
                )

        if pendientes:
            try:
                # Segunda red: descarta lo que sea casi idéntico a lo que ya hay.
                memory_manager.fusionar(state['usuario'], pendientes)
            except Exception:  # noqa: BLE001 - idem: Chroma o la red pueden fallar
                return {}
        return {}

    grafo = StateGraph(EstadoChat)
    grafo.add_node('recuperar', nodo_recuperar)
    grafo.add_node('responder', nodo_responder)
    grafo.add_node('aprender', nodo_aprender)

    # Aprender primero: el hecho del mensaje actual entra en la memoria que se
    # recupera y se inyecta en la misma respuesta, no en la siguiente.
    grafo.add_edge(START, 'aprender')
    grafo.add_edge('aprender', 'recuperar')
    grafo.add_edge('recuperar', 'responder')
    grafo.add_edge('responder', END)

    return grafo.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Instancia para la aplicación
# ---------------------------------------------------------------------------
_APP: Any = None
_SAVER: SqliteSaver | None = None


def _checkpointer() -> SqliteSaver:
    """Checkpointer SQLite, uno por proceso y en su propio fichero."""
    global _SAVER
    if _SAVER is None:
        conexion = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        _SAVER = SqliteSaver(conexion)
    return _SAVER


def app() -> Any:
    """Grafo compilado que usa la aplicación (se construye una sola vez)."""
    global _APP
    if _APP is None:
        _APP = construir(checkpointer=_checkpointer())
    return _APP


def _config(chat_id: str) -> dict[str, Any]:
    """Configuración del hilo: el chat es el `thread_id` del checkpointer."""
    return {'configurable': {'thread_id': chat_id}}


def _entrada(usuario: str, chat_id: str, texto: str) -> dict[str, Any]:
    """Estado inicial de un turno."""
    return {
        'mensajes': [HumanMessage(content=texto)],
        'usuario': usuario,
        'chat': chat_id,
        'memoria': '',
    }


def responder(usuario: str, chat_id: str, texto: str) -> str:
    """Responde al mensaje y deja el turno guardado (sin streaming)."""
    if not texto.strip():
        raise ValueError('el mensaje no puede estar vacío')

    resultado = app().invoke(_entrada(usuario, chat_id, texto), _config(chat_id))
    mensajes = resultado.get('mensajes') or []
    return str(mensajes[-1].content) if mensajes else ''


def responder_stream(usuario: str, chat_id: str, texto: str) -> Iterator[str]:
    """Igual que `responder`, pero va soltando los trozos según llegan.

    Solo se emiten los tokens del nodo ``responder``; el extractor de memoria
    también usa el modelo y sus tokens no deben verse en pantalla.
    """
    if not texto.strip():
        raise ValueError('el mensaje no puede estar vacío')

    for trozo, metadatos in app().stream(
        _entrada(usuario, chat_id, texto),
        _config(chat_id),
        stream_mode='messages',
    ):
        if metadatos.get('langgraph_node') != 'responder':
            continue
        contenido = trozo.content
        if isinstance(contenido, str) and contenido:
            yield contenido


def historial(chat_id: str) -> list[BaseMessage]:
    """Conversación guardada de ese chat, para pintarla al reabrirlo."""
    estado = app().get_state(_config(chat_id))
    if not estado.values:
        return []
    return list(estado.values.get('mensajes') or [])
