"""Chats de cada usuario.

Aquí se guarda **solo la ficha** del chat: a quién pertenece, su título y las
fechas. La conversación en sí vive en el checkpointer de LangGraph, un hilo por
chat (`thread_id = id del chat`), que es la memoria de contexto.

Responsabilidad de este módulo: crear, listar, renombrar y borrar chats, y
mantener el título automático y la fecha de última actividad.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime

import db

# Longitud del título automático que se genera con el primer mensaje.
MAX_TITULO = 40


@dataclass(frozen=True)
class Chat:
    """Ficha de una conversación, tal y como se pinta en la lista."""

    id: str
    usuario_id: int
    titulo: str
    creado_en: str
    actualizado_en: str


def nuevo_id() -> str:
    """Identificador ordenable y sin colisiones: `20261120-1830-a1b2`."""
    marca = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
    return f'{marca}-{uuid.uuid4().hex[:4]}'


def titulo_automatico(texto: str) -> str:
    """Título a partir del primer mensaje: una línea y sin cortar palabras.

    Devuelve cadena vacía si el texto no aporta nada, para no poner títulos
    absurdos a un mensaje en blanco.
    """
    limpio = ' '.join(texto.split())
    if not limpio:
        return ''
    if len(limpio) <= MAX_TITULO:
        return limpio

    recorte = limpio[:MAX_TITULO]
    # Se corta por la última palabra completa, si hay alguna cerca.
    if ' ' in recorte:
        recorte = recorte.rsplit(' ', 1)[0]
    return recorte.rstrip(' ,.;:') + '…'


def _a_chat(fila: sqlite3.Row) -> Chat:
    """Convierte una fila de la base de datos en el objeto de la interfaz."""
    return Chat(
        id=str(fila['id']),
        usuario_id=int(fila['usuario_id']),
        titulo=str(fila['titulo']),
        creado_en=str(fila['creado_en']),
        actualizado_en=str(fila['actualizado_en']),
    )


def crear(usuario_id: int, *, titulo: str = '') -> Chat:
    """Abre un chat vacío para ese usuario.

    Raises:
        ValueError: si el usuario no existe (la clave foránea lo rechaza).
    """
    identificador = nuevo_id()
    marca = db.ahora()
    with db.sesion() as conexion:
        try:
            conexion.execute(
                """
                INSERT INTO chats (id, usuario_id, titulo, creado_en, actualizado_en)
                VALUES (?, ?, ?, ?, ?)
                """,
                (identificador, usuario_id, titulo.strip(), marca, marca),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f'no existe el usuario con id {usuario_id}'
            ) from exc

    return Chat(
        id=identificador,
        usuario_id=usuario_id,
        titulo=titulo.strip(),
        creado_en=marca,
        actualizado_en=marca,
    )


def listar(usuario_id: int) -> list[Chat]:
    """Chats del usuario, del más reciente al más antiguo."""
    with db.sesion() as conexion:
        filas = conexion.execute(
            'SELECT * FROM chats WHERE usuario_id = ? ORDER BY actualizado_en DESC',
            (usuario_id,),
        ).fetchall()
    return [_a_chat(fila) for fila in filas]


def obtener(chat_id: str) -> Chat | None:
    """Busca un chat por su id, o ``None`` si no existe."""
    with db.sesion() as conexion:
        fila = conexion.execute(
            'SELECT * FROM chats WHERE id = ?', (chat_id,)
        ).fetchone()
    return _a_chat(fila) if fila else None


def renombrar(chat_id: str, titulo: str) -> Chat:
    """Cambia el título a mano.

    Raises:
        ValueError: si el chat no existe o el título está vacío.
    """
    limpio = ' '.join(titulo.split())
    if not limpio:
        raise ValueError('el título no puede estar vacío')

    chat = obtener(chat_id)
    if chat is None:
        raise ValueError(f'no existe el chat {chat_id!r}')

    with db.sesion() as conexion:
        conexion.execute(
            'UPDATE chats SET titulo = ? WHERE id = ?', (limpio, chat_id)
        )

    return Chat(
        id=chat.id,
        usuario_id=chat.usuario_id,
        titulo=limpio,
        creado_en=chat.creado_en,
        actualizado_en=chat.actualizado_en,
    )


def registrar_mensaje(chat_id: str, texto_usuario: str) -> Chat:
    """Actualiza la ficha del chat tras un mensaje del usuario.

    Marca la actividad y, si el chat aún no tiene título, lo genera con el
    primer mensaje. La conversación no se guarda aquí: eso lo hace el grafo en
    su checkpointer.

    Raises:
        ValueError: si el chat no existe.
    """
    chat = obtener(chat_id)
    if chat is None:
        raise ValueError(f'no existe el chat {chat_id!r}')

    titulo = chat.titulo or titulo_automatico(texto_usuario)
    marca = db.ahora()
    with db.sesion() as conexion:
        conexion.execute(
            'UPDATE chats SET titulo = ?, actualizado_en = ? WHERE id = ?',
            (titulo, marca, chat_id),
        )

    return Chat(
        id=chat.id,
        usuario_id=chat.usuario_id,
        titulo=titulo,
        creado_en=chat.creado_en,
        actualizado_en=marca,
    )


def borrar(chat_id: str) -> None:
    """Borra el chat y su conversación del checkpointer.

    Primero la conversación: si algo falla, la ficha sigue ahí y el chat no
    desaparece de la lista a medias.

    Raises:
        ValueError: si el chat no existe.
    """
    chat = obtener(chat_id)
    if chat is None:
        raise ValueError(f'no existe el chat {chat_id!r}')

    db.borrar_hilos([chat.id])
    with db.sesion() as conexion:
        conexion.execute('DELETE FROM chats WHERE id = ?', (chat.id,))
