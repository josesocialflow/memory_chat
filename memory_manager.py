"""Memoria transversal de un usuario, guardada en Chroma.

Qué es
------
Los datos que valen para **todos** los chats de una persona: cómo se llama,
dónde trabaja, qué prefiere, hechos que ha contado. No confundir con la memoria
de contexto de un chat concreto, que vive en el checkpointer del grafo.

Cómo se organiza
----------------
Una colección de Chroma por usuario (`memoria_<slug>`), dentro de `data/chroma`.
El aislamiento es estructural: una consulta no puede devolver recuerdos de otra
persona aunque se olvide un filtro.

Cada recuerdo guarda:
    documents  -> el texto del recuerdo (lo que se vectoriza)
    metadatas  -> categoria, origen_chat, creado_en, actualizado_en, activo

`activo` permite sustituir un recuerdo sin borrarlo: el viejo se marca como
inactivo y deja de aparecer en las consultas, pero queda el rastro de qué se
sabía antes.

Este módulo no llama al modelo de lenguaje. La extracción de candidatos es cosa
del grafo; aquí solo se decide qué hacer con ellos (`fusionar`).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction, Metadata
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions

from config import (
    CATEGORIAS_SIEMPRE,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    HNSW_SPACE,
    MAX_MEMORY_RESULTS,
    MEMORY_CATEGORIES,
    UMBRAL_DUPLICADO,
    exigir_api_key,
)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recuerdo:
    """Un dato del usuario que sobrevive a los chats."""

    id: str
    categoria: str
    texto: str
    origen_chat: str = ''
    creado_en: str = ''
    actualizado_en: str = ''
    activo: bool = True
    # Solo se rellena al recuperar: distancia al texto consultado (coseno).
    distancia: float | None = None


@dataclass(frozen=True)
class Candidato:
    """Recuerdo propuesto por el extractor, antes de decidir si entra."""

    categoria: str
    texto: str
    origen_chat: str = ''


@dataclass(frozen=True)
class ResultadoFusion:
    """Qué ha pasado al fusionar una lista de candidatos."""

    nuevos: list[Recuerdo]
    repetidos: list[str]


# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------
_CLIENTE: ClientAPI | None = None
_FUNCION: EmbeddingFunction | None = None


def ahora() -> str:
    """Marca temporal local en ISO, con segundos y desfase horario."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def cliente() -> ClientAPI:
    """Cliente persistente de Chroma, uno por proceso.

    Abrir dos clientes sobre la misma carpeta da problemas, así que se crea una
    sola vez y se reutiliza. La telemetría de Chroma se desactiva: no aporta
    nada y añade tráfico en cada arranque.
    """
    global _CLIENTE
    if _CLIENTE is None:
        _CLIENTE = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _CLIENTE


def funcion_embeddings() -> EmbeddingFunction:
    """Función de embeddings de OpenAI, cacheada.

    Se pasa explícitamente en cada apertura de colección: Chroma guarda su
    configuración y trata de reconstruirla, y con funciones propias eso acaba en
    avisos. El modelo y el nombre de la variable de entorno se fuerzan porque
    los valores por defecto de Chroma no son los de este proyecto.
    """
    global _FUNCION
    if _FUNCION is None:
        _FUNCION = embedding_functions.OpenAIEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            api_key=exigir_api_key(),
        )
    return _FUNCION


def usar_cliente_para_tests(
    cliente_nuevo: ClientAPI,
    funcion: EmbeddingFunction | None = None,
) -> None:
    """Sustituye el cliente y, si se pasa, la función de embeddings.

    Pensado para los tests: permite usar `EphemeralClient` y un embedding falso
    sin tocar la configuración ni llamar a OpenAI.
    """
    global _CLIENTE, _FUNCION
    _CLIENTE = cliente_nuevo
    _FUNCION = funcion


# ---------------------------------------------------------------------------
# Colecciones
# ---------------------------------------------------------------------------
def nombre_coleccion(slug: str) -> str:
    """Nombre de colección válido en Chroma a partir del slug del usuario.

    Chroma exige 3-63 caracteres, minúsculas, letras/números/`_`/`-` y empezar y
    terminar por carácter alfanumérico. El slug lo genera `users.py`; aquí solo
    se comprueba para fallar con un mensaje claro en vez de con un error opaco
    de Chroma.
    """
    nombre = f'memoria_{slug.strip().lower()}'
    patron = r'memoria_[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?'
    if len(nombre) > 63 or not re.fullmatch(patron, nombre):
        raise ValueError(f'slug no válido para una colección de Chroma: {slug!r}')
    return nombre


def coleccion(slug: str) -> chromadb.Collection:
    """Abre (o crea) la colección de memoria de ese usuario."""
    return cliente().get_or_create_collection(
        name=nombre_coleccion(slug),
        embedding_function=funcion_embeddings(),
        metadata={'hnsw:space': HNSW_SPACE},
    )


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _validar(categoria: str, texto: str) -> None:
    """Comprueba categoría conocida y texto no vacío."""
    if categoria not in MEMORY_CATEGORIES:
        raise ValueError(
            f'categoría no válida: {categoria!r}; se admite {MEMORY_CATEGORIES}'
        )
    if not texto.strip():
        raise ValueError('el recuerdo no puede estar vacío')


def _identificador(categoria: str, texto: str) -> str:
    """Id estable: el mismo hecho, escrito igual, no se duplica nunca."""
    normalizado = ' '.join(texto.casefold().split())
    return hashlib.sha1(f'{categoria}|{normalizado}'.encode()).hexdigest()[:16]


def _metadatos(
    categoria: str,
    origen_chat: str,
    creado_en: str,
    actualizado_en: str,
    *,
    activo: bool = True,
) -> dict[str, Any]:
    """Metadatos con los tipos que admite Chroma: nada de None ni listas."""
    return {
        'categoria': categoria,
        'origen_chat': origen_chat,
        'creado_en': creado_en,
        'actualizado_en': actualizado_en,
        'activo': activo,
    }


def _recuerdo(
    identificador: str,
    texto: str,
    metadatos: Mapping[str, Any] | None,
    distancia: float | None = None,
) -> Recuerdo:
    """Convierte lo que devuelve Chroma en un `Recuerdo`."""
    datos = metadatos or {}
    return Recuerdo(
        id=identificador,
        categoria=str(datos.get('categoria', '')),
        texto=texto,
        origen_chat=str(datos.get('origen_chat', '')),
        creado_en=str(datos.get('creado_en', '')),
        actualizado_en=str(datos.get('actualizado_en', '')),
        activo=bool(datos.get('activo', True)),
        distancia=distancia,
    )


def _filtro(
    *, activo: bool | None = True, categoria: str | None = None
) -> dict[str, Any] | None:
    """Compone el `where` de Chroma a partir de cero, uno o dos criterios."""
    clausulas: list[dict[str, Any]] = []
    if activo is not None:
        clausulas.append({'activo': activo})
    if categoria is not None:
        clausulas.append({'categoria': categoria})

    if not clausulas:
        return None
    if len(clausulas) == 1:
        return clausulas[0]
    return {'$and': clausulas}


# ---------------------------------------------------------------------------
# Altas y cambios
# ---------------------------------------------------------------------------
def agregar(
    slug: str,
    categoria: str,
    texto: str,
    origen_chat: str = '',
) -> Recuerdo:
    """Guarda un recuerdo. Si ya existe ese mismo hecho, no lo duplica.

    El id se calcula a partir de categoría y texto, así que repetir el mismo
    alta es idempotente. Si el recuerdo existía pero estaba inactivo (se
    sustituyó en el pasado), se reactiva: el hecho ha vuelto a ser cierto.
    """
    _validar(categoria, texto)
    texto = texto.strip()
    base = coleccion(slug)
    identificador = _identificador(categoria, texto)

    existentes = base.get(ids=[identificador])
    if existentes['ids']:
        metadatos = (existentes['metadatas'] or [None])[0]
        documento = (existentes['documents'] or [''])[0]
        if metadatos and not bool(metadatos.get('activo', True)):
            reactivado = _metadatos(
                categoria,
                origen_chat or str(metadatos.get('origen_chat', '')),
                str(metadatos.get('creado_en', ahora())),
                ahora(),
            )
            base.update(ids=[identificador], metadatas=[reactivado])
            return _recuerdo(identificador, documento or texto, reactivado)
        return _recuerdo(identificador, documento or texto, metadatos)

    marca = ahora()
    base.upsert(
        ids=[identificador],
        documents=[texto],
        metadatas=[_metadatos(categoria, origen_chat, marca, marca)],
    )
    return Recuerdo(
        id=identificador,
        categoria=categoria,
        texto=texto,
        origen_chat=origen_chat,
        creado_en=marca,
        actualizado_en=marca,
    )


def sustituir_varios(
    slug: str,
    ids_viejos: Iterable[str],
    texto_nuevo: str,
    *,
    categoria: str | None = None,
    origen_chat: str = '',
) -> Recuerdo:
    """Reemplaza uno o varios recuerdos por un único hecho nuevo.

    Se usa cuando un hecho invalida varios anteriores a la vez: un cambio de
    empleo deja obsoletos el puesto anterior **y** los datos que dependían de
    él (zonas, clientes, departamento). Todos se marcan inactivos y se guarda
    un solo recuerdo nuevo.

    No se borra nada a propósito: queda el rastro de qué se sabía antes y
    cuándo cambió.
    """
    base = coleccion(slug)
    ids = list(dict.fromkeys(ids_viejos))
    if not ids:
        raise ValueError('no hay recuerdos que sustituir')

    datos = base.get(ids=ids)
    encontrados: dict[str, tuple[str, Mapping[str, Any]]] = {
        identificador: (documento or '', metadatos or {})
        for identificador, documento, metadatos in zip(
            datos['ids'],
            datos['documents'] or [],
            datos['metadatas'] or [],
            strict=False,
        )
    }
    faltan = [identificador for identificador in ids if identificador not in encontrados]
    if faltan:
        raise ValueError(f'no existen los recuerdos {faltan!r}')

    primer_metadato = encontrados[ids[0]][1]
    categoria_final = categoria or str(primer_metadato.get('categoria', ''))
    origen = origen_chat or str(primer_metadato.get('origen_chat', ''))
    _validar(categoria_final, texto_nuevo)
    texto_nuevo = texto_nuevo.strip()

    marca = ahora()
    metadatos_inactivos: list[Metadata] = [
        _metadatos(
            str(encontrados[identificador][1].get('categoria', categoria_final)),
            str(encontrados[identificador][1].get('origen_chat', origen)),
            str(encontrados[identificador][1].get('creado_en', marca)),
            marca,
            activo=False,
        )
        for identificador in ids
    ]
    base.update(ids=ids, metadatas=metadatos_inactivos)

    # `agregar` crea el recuerdo nuevo o reactiva uno inactivo con el mismo
    # texto (por ejemplo, si el usuario vuelve a un empleo anterior).
    return agregar(slug, categoria_final, texto_nuevo, origen)


def sustituir(
    slug: str,
    id_viejo: str,
    texto_nuevo: str,
    *,
    categoria: str | None = None,
    origen_chat: str = '',
) -> Recuerdo:
    """Reemplaza un recuerdo; atajo de `sustituir_varios` para un solo id."""
    return sustituir_varios(
        slug,
        [id_viejo],
        texto_nuevo,
        categoria=categoria,
        origen_chat=origen_chat,
    )


def borrar(slug: str, id_recuerdo: str) -> None:
    """Borra un recuerdo de verdad (para la pantalla de memoria)."""
    coleccion(slug).delete(ids=[id_recuerdo])


def borrar_todo(slug: str) -> None:
    """Elimina la colección completa del usuario.

    Tolera que no exista: al dar de baja a alguien que nunca habló, la colección
    no se llegó a crear.
    """
    try:
        cliente().delete_collection(nombre_coleccion(slug))
    except NotFoundError:
        return


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------
def total(slug: str) -> int:
    """Cuántos recuerdos tiene el usuario (activos o no)."""
    return coleccion(slug).count()


def listar(
    slug: str,
    categoria: str | None = None,
    *,
    incluir_inactivos: bool = False,
) -> list[Recuerdo]:
    """Recuerdos del usuario, del más antiguo al más nuevo.

    La pantalla de memoria usa `incluir_inactivos=True` para poder enseñar el
    histórico de lo sustituido.
    """
    base = coleccion(slug)
    filtro = _filtro(
        activo=None if incluir_inactivos else True,
        categoria=categoria,
    )
    datos = base.get(where=filtro) if filtro else base.get()

    recuerdos = [
        _recuerdo(identificador, documento or '', metadatos)
        for identificador, documento, metadatos in zip(
            datos['ids'],
            datos['documents'] or [],
            datos['metadatas'] or [],
            strict=False,
        )
    ]
    recuerdos.sort(key=lambda recuerdo: (recuerdo.creado_en, recuerdo.id))
    return recuerdos


def recuperar(
    slug: str,
    consulta: str,
    maximo: int = MAX_MEMORY_RESULTS,
    *,
    categoria: str | None = None,
) -> list[Recuerdo]:
    """Los recuerdos más parecidos al texto, hasta `maximo`.

    Devuelve lista vacía si el usuario no tiene memoria todavía: no es un error,
    es alguien nuevo.
    """
    base = coleccion(slug)
    disponibles = base.count()
    if disponibles == 0 or not consulta.strip() or maximo <= 0:
        return []

    resultado = base.query(
        query_texts=[consulta],
        n_results=min(maximo, disponibles),
        where=_filtro(categoria=categoria),
    )

    distancias = (resultado.get('distances') or [[]])[0]
    return [
        _recuerdo(identificador, documento or '', metadatos, distancia)
        for identificador, documento, metadatos, distancia in zip(
            resultado['ids'][0],
            (resultado.get('documents') or [[]])[0],
            (resultado.get('metadatas') or [[]])[0],
            distancias if distancias else [None] * len(resultado['ids'][0]),
            strict=False,
        )
    ]


def recuperar_contexto(
    slug: str,
    consulta: str,
    maximo: int = MAX_MEMORY_RESULTS,
    *,
    categorias_fijas: Iterable[str] = CATEGORIAS_SIEMPRE,
) -> list[Recuerdo]:
    """Recuerdos que se inyectan en el prompt de este turno.

    Es una mezcla de dos criterios, y el orden importa:

    1. **Siempre** los activos de las categorías que definen a la persona
       (`CATEGORIAS_SIEMPRE`: personal, profesional y hechos importantes). Son
       pocos y no pueden depender del parecido con la pregunta: si solo se
       buscara por similitud, "¿a qué me dedico?" traería el trabajo antiguo,
       que se parece más a la pregunta que el nuevo.
    2. Hasta `maximo` recuerdos **extra** por similitud con el mensaje, de las
       categorías restantes (preferencias, sobre todo). Así una pregunta sobre
       viajes puede traer las preferencias de viaje sin desplazar los hechos
       personales y profesionales.

    Los fijos se ordenan del más reciente al más antiguo dentro de su categoría,
    y nunca se repite un id entre los dos grupos.
    """
    fijas = tuple(categorias_fijas)
    resultado: list[Recuerdo] = []
    vistos: set[str] = set()

    for categoria in fijas:
        recientes = sorted(
            listar(slug, categoria),
            key=lambda recuerdo: (recuerdo.actualizado_en, recuerdo.id),
            reverse=True,
        )
        for recuerdo in recientes:
            if recuerdo.id not in vistos:
                vistos.add(recuerdo.id)
                resultado.append(recuerdo)

    if maximo <= 0 or not consulta.strip():
        return resultado

    # Se piden de más porque parte de los resultados volverá a caer en las
    # categorías fijas, que ya están cubiertas: así quedan huecos para los
    # extras de verdad.
    for recuerdo in recuperar(slug, consulta, maximo=maximo + len(resultado)):
        if recuerdo.id in vistos or recuerdo.categoria in fijas:
            continue
        vistos.add(recuerdo.id)
        resultado.append(recuerdo)
        extras = sum(1 for item in resultado if item.categoria not in fijas)
        if extras >= maximo:
            break

    return resultado


# ---------------------------------------------------------------------------
# Fusión de candidatos
# ---------------------------------------------------------------------------
def fusionar(slug: str, candidatos: Iterable[Candidato]) -> ResultadoFusion:
    """Decide qué hacer con los recuerdos que propone el extractor.

    Regla:

    - Si el candidato es **muy parecido** a algo que ya está en su misma
      categoría (distancia por debajo de `UMBRAL_DUPLICADO`), se descarta como
      repetido.
    - Si es distinto, se guarda.

    **No detecta contradicciones automáticamente**: un cambio de trabajo es un
    recuerdo nuevo y el anterior sigue activo hasta que el extractor rellene
    `sustituye_a` (lo normal) o se sustituya a mano desde la pantalla de
    memoria. Es una decisión deliberada: adivinar contradicciones por distancia
    produce sustituciones falsas, porque los pares "mismo hecho" y "hecho
    distinto" se solapan en distancia.
    """
    nuevos: list[Recuerdo] = []
    repetidos: list[str] = []

    for candidato in candidatos:
        _validar(candidato.categoria, candidato.texto)
        parecidos = recuperar(
            slug,
            candidato.texto,
            maximo=1,
            categoria=candidato.categoria,
        )
        distancia = parecidos[0].distancia if parecidos else None
        if distancia is not None and distancia <= UMBRAL_DUPLICADO:
            repetidos.append(candidato.texto)
            continue

        nuevos.append(
            agregar(slug, candidato.categoria, candidato.texto, candidato.origen_chat)
        )

    return ResultadoFusion(nuevos=nuevos, repetidos=repetidos)


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------
def formatear(recuerdos: Iterable[Recuerdo]) -> str:
    """Bloque de texto para inyectar en el prompt, agrupado por categoría.

    Devuelve cadena vacía si no hay nada: así el llamador no tiene que decidir
    si añadir un encabezado vacío.
    """
    por_categoria: dict[str, list[str]] = {}
    for recuerdo in recuerdos:
        por_categoria.setdefault(recuerdo.categoria, []).append(recuerdo.texto)

    lineas: list[str] = []
    for categoria in MEMORY_CATEGORIES:
        textos = por_categoria.pop(categoria, [])
        if textos:
            lineas.append(f'- {categoria}: ' + '; '.join(textos))
    # Categorías desconocidas (datos antiguos): se muestran al final.
    for categoria, textos in por_categoria.items():
        lineas.append(f'- {categoria}: ' + '; '.join(textos))

    if not lineas:
        return ''
    return 'Lo que sé del usuario:\n' + '\n'.join(lineas)
