"""Chat multiusuario con memoria avanzada (Streamlit).

Cuatro pantallas, sin login real (es una demo local):

- **Usuarios**: alta, baja y selector.
- **Mis chats**: lista de conversaciones del usuario y "nuevo chat".
- **Chat**: historial, mensaje nuevo con streaming.
- **Memoria**: recuerdos del usuario por categoría, con alta, edición y borrado.

Toda la conversación vive en el checkpointer de LangGraph y la memoria en Chroma;
este módulo solo pinta y llama a los servicios. El estado de la interfaz está en
`st.session_state`: `usuario_slug`, `chat_id` y `vista`.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from itertools import chain
from typing import Literal

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import chats
import graph
import memory_manager
import users
from config import DEFAULT_MODEL, MEMORY_CATEGORIES, PAGE_ICON, PAGE_TITLE

# Presentación de cada categoría de memoria: icono y color de la insignia.
ColorBadge = Literal[
    'red', 'orange', 'yellow', 'blue', 'green', 'violet', 'gray', 'grey', 'primary'
]
ICONOS_CATEGORIA = {
    'personal': ':material/person:',
    'profesional': ':material/work:',
    'preferencias': ':material/tune:',
    'hechos_importantes': ':material/star:',
}
COLORES_CATEGORIA: dict[str, ColorBadge] = {
    'personal': 'blue',
    'profesional': 'green',
    'preferencias': 'violet',
    'hechos_importantes': 'orange',
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _fecha(iso: str) -> str:
    """Fecha legible a partir del ISO que guardamos en la base de datos."""
    try:
        return datetime.fromisoformat(iso).strftime('%d/%m/%Y %H:%M')
    except ValueError:
        return iso


def _estado() -> None:
    """Valores de sesión con su valor por defecto."""
    st.session_state.setdefault('usuario_slug', '')
    st.session_state.setdefault('chat_id', '')
    st.session_state.setdefault('vista', 'chats')


def _usuario() -> users.Usuario | None:
    """Usuario seleccionado, si sigue existiendo."""
    identificador = st.session_state.get('usuario_slug', '')
    return users.obtener(identificador) if identificador else None


def _ir(vista: str) -> None:
    """Cambia de pantalla y refresca."""
    st.session_state['vista'] = vista
    st.rerun()


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
def _barra_lateral(usuario: users.Usuario | None) -> None:
    """Identidad actual y navegación."""
    with st.sidebar:
        st.subheader(':material/person: Usuario')
        if usuario is None:
            st.caption('Ninguno seleccionado')
        else:
            st.caption(f'**{usuario.nombre}**')
            total_chats = len(chats.listar(usuario.id))
            total_recuerdos = len(memory_manager.listar(usuario.slug))
            st.caption(f'{total_chats} chats · {total_recuerdos} recuerdos')

        st.divider()
        if usuario is not None:
            if st.button(':material/forum: Mis chats', width='stretch'):
                _ir('chats')
            if st.button(':material/psychology: Memoria', width='stretch'):
                _ir('memoria')
            if st.button(':material/switch_account: Cambiar de usuario', width='stretch'):
                st.session_state['usuario_slug'] = ''
                st.session_state['chat_id'] = ''
                _ir('chats')

        st.divider()
        st.caption(
            'La conversación se guarda como memoria de contexto del chat; los '
            'recuerdos de la pestaña Memoria valen para todos los chats.'
        )
        with st.expander(':material/info: Cómo funciona'):
            st.markdown(
                '- **Cada chat** guarda su conversación completa.\n'
                '- **Cada usuario** tiene recuerdos que valen para todos sus chats.\n'
                '- Al escribir, se buscan los recuerdos relacionados y se añaden '
                'al prompt; al responder, se aprenden los nuevos.'
            )


# ---------------------------------------------------------------------------
# Pantalla: usuarios
# ---------------------------------------------------------------------------
def _pantalla_usuarios() -> None:
    """Alta, baja y entrada a un usuario."""
    st.header(':material/group: Usuarios')
    st.caption('Elige con quién quieres conversar, o crea uno nuevo.')

    disponibles = users.listar()
    if disponibles:
        for usuario in disponibles:
            with st.container(border=True):
                izquierda, derecha = st.columns([4, 1])
                izquierda.markdown(f'**{usuario.nombre}**')
                izquierda.caption(
                    f'{len(chats.listar(usuario.id))} chats · '
                    f'última actividad {_fecha(usuario.ultima_actividad)}'
                )
                if derecha.button(
                    'Entrar', key=f'entrar-{usuario.slug}', width='stretch'
                ):
                    st.session_state['usuario_slug'] = usuario.slug
                    st.session_state['chat_id'] = ''
                    _ir('chats')
    else:
        st.info('Todavía no hay usuarios. Crea el primero abajo.', icon=':material/inbox:')

    st.divider()
    with st.form('crear-usuario', clear_on_submit=True):
        st.subheader(':material/person_add: Nuevo usuario')
        nombre = st.text_input('Nombre', placeholder='Ana Pérez')
        crear = st.form_submit_button('Crear usuario')

    if crear:
        try:
            nuevo = users.crear(nombre)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state['usuario_slug'] = nuevo.slug
            st.session_state['chat_id'] = ''
            st.session_state['vista'] = 'chats'
            st.rerun()

    if disponibles:
        with st.expander(':material/delete: Dar de baja un usuario'):
            st.caption(
                'Se borran sus chats, sus conversaciones y su memoria. '
                'No se puede deshacer.'
            )
            elegido = st.selectbox(
                'Usuario', [usuario.slug for usuario in disponibles],
                format_func=lambda valor: next(
                    usuario.nombre for usuario in disponibles if usuario.slug == valor
                ),
                key='baja-usuario',
            )
            confirmar = st.checkbox('Confirmo que quiero borrarlo todo')
            if st.button('Dar de baja', type='primary', disabled=not confirmar):
                try:
                    users.borrar(elegido)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if st.session_state.get('usuario_slug') == elegido:
                        st.session_state['usuario_slug'] = ''
                        st.session_state['chat_id'] = ''
                    st.success('Usuario eliminado')
                    st.rerun()


# ---------------------------------------------------------------------------
# Pantalla: chats
# ---------------------------------------------------------------------------
def _pantalla_chats(usuario: users.Usuario) -> None:
    """Lista de conversaciones y botón para empezar una nueva."""
    st.header(':material/forum: Mis chats')
    st.caption(f'Conversaciones de {usuario.nombre}.')

    if st.button(':material/add_comment: Nuevo chat', type='primary'):
        nuevo = chats.crear(usuario.id)
        st.session_state['chat_id'] = nuevo.id
        _ir('chat')

    lista = chats.listar(usuario.id)
    if not lista:
        st.info(
            'Sin conversaciones todavía. Empieza una con “Nuevo chat”.',
            icon=':material/chat_bubble_outline:',
        )
        return

    for chat in lista:
        with st.container(border=True):
            izquierda, derecha = st.columns([5, 2])
            izquierda.markdown(f'**{chat.titulo or "Chat sin título"}**')
            izquierda.caption(f'Actualizado el {_fecha(chat.actualizado_en)}')
            if derecha.button('Abrir', key=f'abrir-{chat.id}', width='stretch'):
                st.session_state['chat_id'] = chat.id
                _ir('chat')
            if derecha.button('Borrar', key=f'borrar-{chat.id}', width='stretch'):
                try:
                    chats.borrar(chat.id)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if st.session_state.get('chat_id') == chat.id:
                        st.session_state['chat_id'] = ''
                    st.rerun()


# ---------------------------------------------------------------------------
# Pantalla: chat
# ---------------------------------------------------------------------------
def _pintar_historial(mensajes: list[BaseMessage]) -> None:
    """Conversación guardada, con los roles de cada mensaje."""
    for mensaje in mensajes:
        if isinstance(mensaje, HumanMessage) and isinstance(mensaje.content, str):
            with st.chat_message('user', avatar='🧑‍💼'):
                st.markdown(mensaje.content)
        elif isinstance(mensaje, AIMessage) and isinstance(mensaje.content, str):
            with st.chat_message('assistant', avatar='🤖'):
                st.markdown(mensaje.content)


def _historial(chat_id: str) -> list[BaseMessage]:
    """Conversación guardada, tolerando que el checkpointer aún no exista."""
    try:
        return graph.historial(chat_id)
    except Exception as exc:  # noqa: BLE001 - se avisa en la propia pantalla
        st.warning(f'No se pudo leer la conversación: {type(exc).__name__}')
        return []


def _pantalla_chat(usuario: users.Usuario) -> None:
    """Conversación abierta: historial, entrada y respuesta en streaming."""
    chat = chats.obtener(st.session_state.get('chat_id', ''))
    if chat is None:
        st.info('Ese chat ya no existe. Elige otro.', icon=':material/help:')
        if st.button('Volver a mis chats'):
            _ir('chats')
        return

    cabecera, acciones = st.columns([4, 1])
    cabecera.header(f':material/chat_bubble: {chat.titulo or "Chat nuevo"}')
    cabecera.caption(f'Creado el {_fecha(chat.creado_en)}')
    if acciones.button(':material/arrow_back: Chats', width='stretch'):
        _ir('chats')

    with st.expander(':material/edit: Cambiar el título'):
        titulo = st.text_input('Título', value=chat.titulo, key=f'titulo-{chat.id}')
        if st.button('Guardar título'):
            try:
                chats.renombrar(chat.id, titulo)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    mensajes = _historial(chat.id)
    _pintar_historial(mensajes)
    if not mensajes:
        st.info(
            'Escribe el primer mensaje. Lo que cuentes de ti quedará en tu '
            'memoria y valdrá para todos tus chats.',
            icon=':material/waving_hand:',
        )

    texto = st.chat_input('Escribe tu mensaje')
    if texto:
        try:
            chats.registrar_mensaje(chat.id, texto)
            users.tocar(usuario.slug)
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.chat_message('user', avatar='🧑‍💼'):
            st.markdown(texto)
        with st.chat_message('assistant', avatar='🤖'):
            generador = graph.responder_stream(usuario.slug, chat.id, texto)
            try:
                # El spinner gira durante `aprender` y `recuperar`: el primer
                # `next` es donde se hace ese trabajo y todavía no hay tokens.
                with st.spinner('Actualizando memoria…'):
                    primer_trozo = next(generador, None)
                if primer_trozo is not None:
                    # Se devuelve el trozo ya consumido al frente del stream.
                    st.write_stream(chain([primer_trozo], generador))
            except Exception as exc:  # noqa: BLE001 - se muestra al usuario
                st.error(f'No se pudo responder: {type(exc).__name__}: {exc}')
                return
        st.rerun()


# ---------------------------------------------------------------------------
# Pantalla: memoria
# ---------------------------------------------------------------------------
def _pantalla_memoria(usuario: users.Usuario) -> None:
    """Recuerdos del usuario, por categoría."""
    st.header(':material/psychology: Memoria')
    st.caption(
        f'Lo que se sabe de {usuario.nombre}. Vale para todos sus chats: se '
        'inyecta en el prompt cuando tiene que ver con lo que pregunta.'
    )

    with st.form('nuevo-recuerdo', clear_on_submit=True):
        categoria = st.selectbox('Categoría', MEMORY_CATEGORIES)
        texto = st.text_area('Recuerdo', placeholder='Trabaja en logística')
        guardar = st.form_submit_button('Añadir recuerdo')

    if guardar:
        try:
            memory_manager.agregar(usuario.slug, categoria, texto)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    incluir_inactivos = st.checkbox('Mostrar también los sustituidos')
    pestanas = st.tabs([categoria.capitalize() for categoria in MEMORY_CATEGORIES])

    for pestana, categoria in zip(pestanas, MEMORY_CATEGORIES, strict=True):
        with pestana:
            recuerdos = memory_manager.listar(
                usuario.slug, categoria, incluir_inactivos=incluir_inactivos
            )
            if not recuerdos:
                st.caption('Nada guardado en esta categoría.')
                continue

            for recuerdo in recuerdos:
                with st.container(border=True):
                    st.badge(
                        categoria,
                        icon=ICONOS_CATEGORIA[categoria],
                        color=COLORES_CATEGORIA[categoria],
                    )
                    st.markdown(recuerdo.texto)
                    if not recuerdo.activo:
                        st.caption('_Sustituido por una versión más reciente._')
                    st.caption(
                        f'Desde {_fecha(recuerdo.creado_en)}'
                        + (
                            f' · chat de origen {recuerdo.origen_chat}'
                            if recuerdo.origen_chat
                            else ''
                        )
                    )

                    editar, borrar = st.columns([1, 1])
                    with editar.popover('Editar'):
                        nuevo = st.text_area(
                            'Texto', value=recuerdo.texto, key=f'texto-{recuerdo.id}'
                        )
                        if st.button('Guardar', key=f'guardar-{recuerdo.id}'):
                            try:
                                memory_manager.sustituir(
                                    usuario.slug, recuerdo.id, nuevo
                                )
                            except ValueError as exc:
                                st.error(str(exc))
                            else:
                                st.rerun()
                    if borrar.button('Borrar', key=f'eliminar-{recuerdo.id}'):
                        memory_manager.borrar(usuario.slug, recuerdo.id)
                        st.rerun()


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
def main() -> None:
    """Cabecera, barra lateral y pantalla elegida."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout='wide')
    _estado()

    st.title('Memory Multi-User Chat')
    insignias = st.columns([2, 2, 2, 6])
    insignias[0].badge('memoria en Chroma', icon=':material/memory:', color='violet')
    insignias[1].badge('streaming', icon=':material/bolt:', color='blue')
    insignias[2].badge(DEFAULT_MODEL, icon=':material/smart_toy:', color='gray')
    st.caption(
        'Cada usuario tiene sus chats y una memoria que vale para todos ellos.'
    )

    usuario = _usuario()
    _barra_lateral(usuario)

    if usuario is None:
        _pantalla_usuarios()
        return

    vista = st.session_state.get('vista', 'chats')
    if vista == 'memoria':
        _pantalla_memoria(usuario)
    elif vista == 'chat':
        _pantalla_chat(usuario)
    else:
        _pantalla_chats(usuario)


if __name__ == '__main__':
    main()
