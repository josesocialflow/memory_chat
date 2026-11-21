"""Alta, baja y consulta de usuarios.

Un usuario son dos cosas: una fila en la tabla `usuarios` (SQLite) y una
colección de memoria en Chroma (`memoria_<slug>`). Este módulo es el único que
crea y destruye usuarios, y se encarga de que no quede basura en ninguna de las
dos partes.

No hay contraseñas: la app es local y el selector de usuario sirve para separar
memorias y conversaciones, no para autenticar.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass

import db
import memory_manager


@dataclass(frozen=True)
class Usuario:
    """Lo que se muestra en el selector de usuarios."""

    id: int
    nombre: str
    slug: str
    creado_en: str
    ultima_actividad: str


def slug(nombre: str) -> str:
    """Convierte un nombre en un identificador seguro.

    Vale para dos destinos a la vez: carpeta y nombre de colección de Chroma.
    Por eso se quitan acentos, se pasa a minúsculas y todo lo que no sea letra o
    número se convierte en guion. El máximo es 55 caracteres porque Chroma
    compone `memoria_<slug>` y el límite es 63.
    """
    descompuesto = unicodedata.normalize('NFKD', nombre)
    sin_acentos = ''.join(
        caracter for caracter in descompuesto if not unicodedata.combining(caracter)
    )
    limpio = re.sub(r'[^a-z0-9]+', '-', sin_acentos.lower()).strip('-')
    if not 3 <= len(limpio) <= 55:
        raise ValueError(
            'el nombre debe tener entre 3 y 55 caracteres utilizables '
            '(letras, números, espacios, guiones)'
        )
    return limpio


def _a_usuario(fila: sqlite3.Row) -> Usuario:
    """Convierte una fila de la base de datos en el objeto de la interfaz."""
    return Usuario(
        id=int(fila['id']),
        nombre=str(fila['nombre']),
        slug=str(fila['slug']),
        creado_en=str(fila['creado_en']),
        ultima_actividad=str(fila['ultima_actividad']),
    )


def crear(nombre: str) -> Usuario:
    """Da de alta un usuario por su nombre visible.

    Raises:
        ValueError: si el nombre no es utilizable o ya existe otro usuario con
            el mismo slug (dos «Ana Pérez» comparten identificador).
    """
    nombre = ' '.join(nombre.split())
    if not nombre:
        raise ValueError('el nombre no puede estar vacío')

    identificador = slug(nombre)
    marca = db.ahora()
    with db.sesion() as conexion:
        try:
            cursor = conexion.execute(
                """
                INSERT INTO usuarios (nombre, slug, creado_en, ultima_actividad)
                VALUES (?, ?, ?, ?)
                """,
                (nombre, identificador, marca, marca),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f'ya existe un usuario con ese nombre ({identificador})'
            ) from exc

        return Usuario(
            id=int(cursor.lastrowid or 0),
            nombre=nombre,
            slug=identificador,
            creado_en=marca,
            ultima_actividad=marca,
        )


def listar() -> list[Usuario]:
    """Todos los usuarios, del más activo recientemente al más antiguo."""
    with db.sesion() as conexion:
        filas = conexion.execute(
            'SELECT * FROM usuarios ORDER BY ultima_actividad DESC, nombre'
        ).fetchall()
    return [_a_usuario(fila) for fila in filas]


def obtener(slug_usuario: str) -> Usuario | None:
    """Busca un usuario por su slug, o ``None`` si no existe."""
    with db.sesion() as conexion:
        fila = conexion.execute(
            'SELECT * FROM usuarios WHERE slug = ?', (slug_usuario,)
        ).fetchone()
    return _a_usuario(fila) if fila else None


def existe(slug_usuario: str) -> bool:
    """Indica si ese slug corresponde a un usuario dado de alta."""
    with db.sesion() as conexion:
        fila = conexion.execute(
            'SELECT 1 FROM usuarios WHERE slug = ?', (slug_usuario,)
        ).fetchone()
    return fila is not None


def tocar(slug_usuario: str) -> None:
    """Actualiza la última actividad (lo llama el chat al enviar un mensaje)."""
    with db.sesion() as conexion:
        conexion.execute(
            'UPDATE usuarios SET ultima_actividad = ? WHERE slug = ?',
            (db.ahora(), slug_usuario),
        )


def borrar(slug_usuario: str) -> None:
    """Da de baja al usuario y todo lo suyo.

    El orden importa: primero la memoria (si Chroma falla, el usuario queda
    intacto), después las conversaciones del checkpointer y, al final, la fila
    del usuario, que arrastra sus chats por `ON DELETE CASCADE`.
    """
    usuario = obtener(slug_usuario)
    if usuario is None:
        raise ValueError(f'no existe el usuario {slug_usuario!r}')

    with db.sesion() as conexion:
        hilos = db.hilos_de_usuario(conexion, usuario.id)

    memory_manager.borrar_todo(usuario.slug)
    db.borrar_hilos(hilos)

    with db.sesion() as conexion:
        conexion.execute('DELETE FROM usuarios WHERE id = ?', (usuario.id,))
