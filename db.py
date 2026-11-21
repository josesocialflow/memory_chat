"""Base de datos de usuarios y chats (SQLite).

Una sola tabla de usuarios y otra de chats. Las conversaciones **no** se guardan
aquí: viven en el checkpointer de LangGraph (`data/checkpoints.sqlite`), que es
la memoria de contexto de cada chat. Aquí solo está lo que necesita la interfaz
para pintar la lista: quién es el usuario y qué chats tiene.

SQLite no aplica las claves foráneas por defecto, así que cada conexión activa
`PRAGMA foreign_keys = ON`. Sin eso, el `ON DELETE CASCADE` sería decorativo.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from config import CHECKPOINT_DB, DB_PATH

ESQUEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    id               INTEGER PRIMARY KEY,
    nombre           TEXT NOT NULL,
    slug             TEXT NOT NULL UNIQUE,
    creado_en        TEXT NOT NULL,
    ultima_actividad TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id             TEXT PRIMARY KEY,
    usuario_id     INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    titulo         TEXT NOT NULL DEFAULT '',
    creado_en      TEXT NOT NULL,
    actualizado_en TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chats_usuario ON chats(usuario_id);
"""


def ahora() -> str:
    """Marca temporal local en ISO, con segundos y desfase horario."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def conectar(ruta: Path | str | None = None) -> sqlite3.Connection:
    """Abre la base de datos y garantiza el esquema.

    La ruta se lee del módulo en cada llamada (no al importar), para que los
    tests puedan apuntar a una base temporal sin tocar la de verdad.
    """
    destino = str(ruta or DB_PATH)
    if destino != ':memory:':
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(destino)
    conexion.row_factory = sqlite3.Row
    conexion.execute('PRAGMA foreign_keys = ON')
    conexion.executescript(ESQUEMA)
    return conexion


@contextmanager
def sesion(ruta: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Conexión lista para usar: confirma al salir y revierte si hay error."""
    conexion = conectar(ruta)
    try:
        yield conexion
        conexion.commit()
    except Exception:
        conexion.rollback()
        raise
    finally:
        conexion.close()


def hilos_de_usuario(conexion: sqlite3.Connection, usuario_id: int) -> list[str]:
    """Identificadores de los chats del usuario, que son sus hilos del grafo."""
    filas = conexion.execute(
        'SELECT id FROM chats WHERE usuario_id = ? ORDER BY creado_en',
        (usuario_id,),
    ).fetchall()
    return [str(fila['id']) for fila in filas]


def borrar_hilos(hilos: Iterable[str]) -> None:
    """Borra del checkpointer las conversaciones de esos chats.

    LangGraph gestiona sus propias tablas, así que se limpian por SQL directo:
    `checkpoints` y `writes` se indexan por `thread_id`. Si las tablas aún no
    existen (nadie ha conversado nunca), no hay nada que borrar.
    """
    identificadores = [hilo for hilo in hilos if hilo]
    if not identificadores or not Path(CHECKPOINT_DB).exists():
        return

    marcas = ', '.join('?' for _ in identificadores)
    conexion = sqlite3.connect(str(CHECKPOINT_DB))
    try:
        for tabla in ('checkpoints', 'writes'):
            try:
                conexion.execute(
                    f'DELETE FROM {tabla} WHERE thread_id IN ({marcas})',
                    identificadores,
                )
            except sqlite3.OperationalError:
                # La tabla todavía no existe: el grafo no se ha usado.
                continue
        conexion.commit()
    finally:
        conexion.close()


if __name__ == '__main__':
    with sesion() as base:
        tablas = [
            fila['name']
            for fila in base.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        usuarios = base.execute('SELECT COUNT(*) FROM usuarios').fetchone()[0]
        chats = base.execute('SELECT COUNT(*) FROM chats').fetchone()[0]

    print(f'Base de datos: {DB_PATH}')
    print('Tablas:', ', '.join(tablas))
    print(f'Usuarios: {usuarios} · chats: {chats}')
